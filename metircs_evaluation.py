from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    # Sort files to ensure renders and GTs align
    render_files = sorted(os.listdir(renders_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    # Check if the number of files match
    if len(render_files) != len(gt_files):
        raise ValueError(f"Mismatch: {len(render_files)} renders vs {len(gt_files)} ground truth images")
    
    for render_fname, gt_fname in zip(render_files, gt_files):
        render_path = renders_dir / render_fname
        gt_path = gt_dir / gt_fname
        if render_fname != gt_fname:
            print(f"Warning: Filename mismatch - Render: {render_fname}, GT: {gt_fname}")
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(render_fname)
    return renders, gts, image_names

def evaluate(model_paths, source_path=None):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            # Define directories
            renders_dir = Path(scene_dir) / "train" / "ours_30000" / "renders" # Training renders from render.py
            if source_path is None:
                # Assume gt_dir is from cfg_args or dataset path; for now, we'll need it provided
                raise ValueError("Please provide the source dataset path with -s to locate ground truth images")
            gt_dir = Path(source_path)  # Ground truth training images from dataset

            # Read training renders and ground truth
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            # Compute metrics
            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

            # Print average metrics
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            # Store results with a method name (e.g., "ours_30000" for consistency)
            method = "ours_30000"  # Assuming iteration 30000 as in your render output
            full_dict[scene_dir][method] = {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item()
            }
            per_view_dict[scene_dir][method] = {
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
            }

            # Save results
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {str(e)}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluate metrics on training renders")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[], help="Path to trained model directory")
    parser.add_argument('--source_path', '-s', type=str, default=None, help="Path to the source dataset with ground truth images")
    args = parser.parse_args()
    evaluate(args.model_paths, args.source_path)