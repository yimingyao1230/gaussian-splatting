from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    render_files = sorted(os.listdir(renders_dir))
    gt_files = sorted(os.listdir(gt_dir))

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

def evaluate(renders_path, gt_path):
    print("\nEvaluating metrics:")
    print("  Renders: ", renders_path)
    print("  GT     : ", gt_path)

    renders_dir = Path(renders_path)
    gt_dir = Path(gt_path)

    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("\n Final Evaluation Results:")
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
    print("")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluate image metrics between rendered and ground truth images")
    parser.add_argument('--renders_path', '-r', required=True, type=str, help="Path to rendered images")
    parser.add_argument('--gt_path', '-g', required=True, type=str, help="Path to ground truth images")
    args = parser.parse_args()

    evaluate(args.renders_path, args.gt_path)