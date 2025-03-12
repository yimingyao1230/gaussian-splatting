#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class OctreeNode:
    def __init__(self, center, size, point_cloud=None, max_points=8, max_depth=8):
        self.center = center
        self.size = size
        self.max_points = max_points
        self.max_depth = max_depth
        self.points = []
        self.children = None
        self.is_leaf = True
        self.point_cloud = point_cloud  # Store reference to the point cloud

    def get_octant_containing_point(self, point):
        """Determine which octant of the current node contains the given point."""
        octant = 0
        if point[0] >= self.center[0]:
            octant |= 4
        if point[1] >= self.center[1]:
            octant |= 2
        if point[2] >= self.center[2]:
            octant |= 1
        return octant

    def split(self):
        """Split the current node into 8 children."""
        self.is_leaf = False
        self.children = []
        half_size = self.size / 2

        # Create 8 children nodes
        for i in range(8):
            child_center = self.center.clone()
            if i & 4:
                child_center[0] += half_size
            else:
                child_center[0] -= half_size
            if i & 2:
                child_center[1] += half_size
            else:
                child_center[1] -= half_size
            if i & 1:
                child_center[2] += half_size
            else:
                child_center[2] -= half_size

            self.children.append(
                OctreeNode(
                    child_center,
                    half_size,
                    self.point_cloud,
                    self.max_points,
                    self.max_depth,
                )
            )

        # Redistribute points to children
        for point_idx in self.points:
            point = self.point_cloud[point_idx]
            octant = self.get_octant_containing_point(point)
            self.children[octant].points.append(point_idx)

        # Clear points from this node
        self.points = []

        # Recursively split children if needed
        for child in self.children:
            if len(child.points) > self.max_points:
                child.split()

    def insert(self, point_idx, point, current_depth=0):
        """Insert a point into the octree."""
        if current_depth == self.max_depth or len(self.points) < self.max_points:
            self.points.append(point_idx)
            return

        if self.is_leaf:
            self.split()

        octant = self.get_octant_containing_point(point)
        self.children[octant].insert(point_idx, point, current_depth + 1)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
        self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float
    ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        print(
            "Number of points before octree compression: ", fused_point_cloud.shape[0]
        )

        # Create temporary features for the initial point cloud
        temp_features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        temp_features[:, :3, 0] = fused_color

        # Calculate initial scales and rotations
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Create temporary opacities
        opacities = self.inverse_opacity_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        # Temporarily set parameters to use octree compression
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        self._features_dc = nn.Parameter(
            temp_features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False)
        )
        self._features_rest = nn.Parameter(
            temp_features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        # Set up temporary radii for octree compression
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Determine appropriate voxel size based on scene scale
        min_coords, _ = torch.min(fused_point_cloud, dim=0)
        max_coords, _ = torch.max(fused_point_cloud, dim=0)
        scene_extent = torch.max(max_coords - min_coords).item()

        # Adjust voxel size based on point cloud density
        # For denser point clouds, use smaller voxel size ratio
        point_count = fused_point_cloud.shape[0]
        if point_count > 1000000:  # Very dense point cloud
            voxel_size_ratio = 200.0
        elif point_count > 500000:  # Dense point cloud
            voxel_size_ratio = 150.0
        elif point_count > 100000:  # Medium density
            voxel_size_ratio = 100.0
        else:  # Sparse point cloud
            voxel_size_ratio = 50.0

        voxel_size = scene_extent / voxel_size_ratio

        print(f"Using voxel size: {voxel_size:.6f} for octree compression")

        # Apply octree compression with fallback - using the special init version
        try:
            compressed_size = self.compress_with_octree_during_init(
                voxel_size=voxel_size, max_points_per_node=8, max_depth=8
            )
            print(f"Number of points after octree compression: {compressed_size}")
        except Exception as e:
            print(f"Error during octree compression: {e}")
            print("Continuing with original point cloud...")
            import traceback

            traceback.print_exc()

        # Now set up the actual parameters for training with the compressed point cloud
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)

        # Initialize the rest of the model
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {
            cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)
        }
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # Clean up temporary variables
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(
                os.path.dirname(path), os.pardir, os.pardir, "exposure.json"
            )
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name])
                    .requires_grad_(False)
                    .cuda()
                    for image_name in exposures
                }
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tmp_radii,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def compress_with_octree(self, voxel_size=0.01, max_points_per_node=8, max_depth=8):
        """
        Compress the point cloud using an octree-based approach.

        Args:
            voxel_size: Size of the smallest octree cell
            max_points_per_node: Maximum number of points in a leaf node before splitting
            max_depth: Maximum depth of the octree
        """
        print("Compressing point cloud with octree...")

        try:
            # Get the bounding box of the point cloud
            xyz = self.get_xyz
            min_coords, _ = torch.min(xyz, dim=0)
            max_coords, _ = torch.max(xyz, dim=0)

            # Calculate the center and size of the root node
            center = (min_coords + max_coords) / 2
            size = torch.max(max_coords - min_coords) / 2

            # Create the octree with a reference to the point cloud
            octree = OctreeNode(center, size, xyz, max_points_per_node, max_depth)

            # Insert all points into the octree
            for i in range(xyz.shape[0]):
                octree.insert(i, xyz[i])

            # Collect leaf nodes for compression
            leaf_nodes = []
            self._collect_leaf_nodes(octree, leaf_nodes)

            print(f"Found {len(leaf_nodes)} leaf nodes in octree")

            # Compress points in each leaf node
            compressed_indices = []
            for node in leaf_nodes:
                if len(node.points) > 1:
                    # Merge points in this node
                    try:
                        merged_idx = self._merge_points_in_node(node)
                        compressed_indices.append(merged_idx)
                    except Exception as e:
                        print(f"Error merging points in node: {e}")
                        # If merging fails, just keep the first point
                        compressed_indices.append(node.points[0])
                elif len(node.points) == 1:
                    # Keep single points as is
                    compressed_indices.append(node.points[0])

            # Create a mask for the points to keep
            keep_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
            for idx in compressed_indices:
                keep_mask[idx] = True

            # Prune the points that are not in the compressed set
            prune_mask = ~keep_mask
            self.prune_points(prune_mask)

            print(
                f"Compressed point cloud from {xyz.shape[0]} to {len(compressed_indices)} points"
            )
            return len(compressed_indices)

        except Exception as e:
            print(f"Error in octree compression: {e}")
            import traceback

            traceback.print_exc()
            return self.get_xyz.shape[0]  # Return original size if compression fails

    def _collect_leaf_nodes(self, node, leaf_nodes):
        """Recursively collect all leaf nodes in the octree."""
        if node.is_leaf and len(node.points) > 0:
            leaf_nodes.append(node)
        elif not node.is_leaf:
            for child in node.children:
                self._collect_leaf_nodes(child, leaf_nodes)

    def _merge_points_in_node(self, node):
        """Merge all points in a leaf node into a single representative point."""
        try:
            point_indices = node.points

            if len(point_indices) <= 1:
                return point_indices[0]  # Nothing to merge

            # Get the attributes of all points in this node
            xyz = self.get_xyz[point_indices]
            features_dc = self._features_dc[point_indices]
            features_rest = self._features_rest[point_indices]
            opacity = self.get_opacity[point_indices]
            scaling = self.get_scaling[point_indices]
            rotation = self.get_rotation[point_indices]

            # Weight by opacity for better visual quality
            # Ensure weights sum to 1 and handle edge cases
            weights = opacity.clone().detach()  # Detach to avoid gradient issues
            weights_sum = weights.sum() + 1e-8  # Add epsilon to avoid division by zero
            weights = weights / weights_sum

            # Compute weighted average for position
            weights_xyz = weights.view(-1, 1)
            avg_xyz = (xyz * weights_xyz).sum(dim=0)

            # Compute weighted average for features
            # For features_dc
            weights_dc = weights.view(-1, 1, 1)
            avg_features_dc = (features_dc * weights_dc).sum(dim=0, keepdim=True)

            # For features_rest
            weights_rest = weights.view(-1, 1, 1)
            avg_features_rest = (features_rest * weights_rest).sum(dim=0, keepdim=True)

            # Compute weighted average for scaling
            avg_scaling = (scaling * weights_xyz).sum(dim=0)

            # Compute weighted average for rotation
            avg_rotation = (rotation * weights_xyz).sum(dim=0)
            avg_rotation = torch.nn.functional.normalize(avg_rotation)

            # Compute max opacity to preserve visibility
            max_opacity = opacity.max()

            # Update the first point with the merged values
            first_idx = point_indices[0]
            self._xyz[first_idx] = avg_xyz
            self._features_dc[first_idx] = avg_features_dc
            self._features_rest[first_idx] = avg_features_rest
            self._opacity[first_idx] = self.inverse_opacity_activation(max_opacity)
            self._scaling[first_idx] = self.scaling_inverse_activation(avg_scaling)
            self._rotation[first_idx] = avg_rotation

            return first_idx

        except Exception as e:
            # If anything goes wrong, just return the first point
            print(f"Error in _merge_points_in_node: {e}")
            import traceback

            traceback.print_exc()
            return node.points[0]

    def prune_points_during_init(self, mask):
        """
        Special version of prune_points to use during initialization before optimizer is created.

        Args:
            mask: Boolean mask of points to remove (True = remove)
        """
        valid_points_mask = ~mask

        # Directly filter the tensors without involving the optimizer
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])

        # Also filter temporary variables if they exist
        if hasattr(self, "xyz_gradient_accum") and self.xyz_gradient_accum.shape[0] > 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        if hasattr(self, "denom") and self.denom.shape[0] > 0:
            self.denom = self.denom[valid_points_mask]

        if hasattr(self, "max_radii2D") and self.max_radii2D.shape[0] > 0:
            self.max_radii2D = self.max_radii2D[valid_points_mask]

        if (
            hasattr(self, "tmp_radii")
            and self.tmp_radii is not None
            and self.tmp_radii.shape[0] > 0
        ):
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def compress_with_octree_during_init(
        self, voxel_size=0.01, max_points_per_node=8, max_depth=8
    ):
        """
        Special version of compress_with_octree to use during initialization before optimizer is created.

        Args:
            voxel_size: Size of the smallest octree cell
            max_points_per_node: Maximum number of points in a leaf node before splitting
            max_depth: Maximum depth of the octree
        """
        print("Compressing point cloud with octree during initialization...")

        try:
            # Get the bounding box of the point cloud
            xyz = self.get_xyz
            min_coords, _ = torch.min(xyz, dim=0)
            max_coords, _ = torch.max(xyz, dim=0)

            # Calculate the center and size of the root node
            center = (min_coords + max_coords) / 2
            size = torch.max(max_coords - min_coords) / 2

            # Create the octree with a reference to the point cloud
            octree = OctreeNode(center, size, xyz, max_points_per_node, max_depth)

            # Insert all points into the octree
            for i in range(xyz.shape[0]):
                octree.insert(i, xyz[i])

            # Collect leaf nodes for compression
            leaf_nodes = []
            self._collect_leaf_nodes(octree, leaf_nodes)

            print(f"Found {len(leaf_nodes)} leaf nodes in octree")

            # Compress points in each leaf node
            compressed_indices = []
            for node in leaf_nodes:
                if len(node.points) > 1:
                    # Merge points in this node
                    try:
                        merged_idx = self._merge_points_in_node(node)
                        compressed_indices.append(merged_idx)
                    except Exception as e:
                        print(f"Error merging points in node: {e}")
                        # If merging fails, just keep the first point
                        compressed_indices.append(node.points[0])
                elif len(node.points) == 1:
                    # Keep single points as is
                    compressed_indices.append(node.points[0])

            # Create a mask for the points to keep
            keep_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
            for idx in compressed_indices:
                keep_mask[idx] = True

            # Prune the points that are not in the compressed set
            prune_mask = ~keep_mask
            self.prune_points_during_init(prune_mask)

            print(
                f"Compressed point cloud from {xyz.shape[0]} to {len(compressed_indices)} points"
            )
            return len(compressed_indices)

        except Exception as e:
            print(f"Error in octree compression during initialization: {e}")
            import traceback

            traceback.print_exc()
            return self.get_xyz.shape[0]  # Return original size if compression fails
