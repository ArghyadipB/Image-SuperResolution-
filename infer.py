"""
Unified Super-Resolution Inference Script

This script performs image super-resolution using one of three models:
EDSR, GAN, or UNet. It supports three operating modes depending on the
provided inputs (LR and/or HR images).
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt

from utils import get_device, psnr, ssim, load_config

# models
from edsr.model import EDSR
from gan.model.generator import Generator
from unet.model import UNET

# ARGUMENT
parser = argparse.ArgumentParser(description="Unified SR Inference")

parser.add_argument("--model", type=str, required=True,
                    choices=["edsr", "gan", "unet"])

parser.add_argument("--lr", type=str, default=None)
parser.add_argument("--hr", type=str, default=None)
parser.add_argument("--output", type=str, default="output.png")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--scale", type=int, default=2)

args = parser.parse_args()

if args.lr is None and args.hr is None:
    raise ValueError("❌ Provide at least --lr or --hr")


# DEVICE
device = get_device(args.device)
print("Using device:", device)


# LOAD MODEL
base_dir = os.path.dirname(__file__)

def load_model():
    if args.model == "edsr":
        config = load_config(os.path.join(base_dir, "edsr/config.yaml"))
        scale = config["model"]["scale"]

        try:
            model = EDSR(
                scale=scale,
                n_resblocks=config["model"]["n_resblocks"],
                n_feats=config["model"]["n_feats"],
                res_scale=config["model"]["res_scale"]
            ).to(device)

            path = os.path.join(base_dir, "edsr/saved_models/edsr.pth")
            model.load_state_dict(torch.load(path, map_location=device))
            print("✅ Loaded EDSR (config)")
            return model, scale

        except:
            model = EDSR(scale=2, n_resblocks=16, n_feats=64, res_scale=0.1).to(device)
            path = os.path.join(base_dir, "edsr/saved_models/edsr.pth")
            model.load_state_dict(torch.load(path, map_location=device))
            print("⚠️ Loaded EDSR fallback")
            return model, 2

    elif args.model == "gan":
        model = Generator().to(device)
        path = os.path.join(base_dir, "gan/saved_models/generator.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        print("✅ Loaded GAN")
        return model, args.scale

    elif args.model == "unet":
        model = UNET().to(device)
        path = os.path.join(base_dir, "unet/saved_models/unet_sr.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        print("✅ Loaded UNet")
        return model, args.scale


model, scale = load_model()
model.eval()


# TRANSFORM
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


# LOAD + STANDARDIZE INPUT
hr_img = None
lr_img_original = None  
lr_img_model = None    

# ---- CASE 1: LR provided ----
if args.lr is not None:
    lr_img_original = Image.open(args.lr).convert("RGB")

    # ALWAYS downsample to 128 for model
    lr_img_model = lr_img_original.resize((128, 128), Image.BICUBIC)

    if args.hr is not None and os.path.exists(args.hr):
        hr_img = Image.open(args.hr).convert("RGB")

# ---- CASE 2: only HR provided ----
elif args.hr is not None:
    hr_img = Image.open(args.hr).convert("RGB")

    # generate LR (128) for model
    lr_img_model = hr_img.resize((128, 128), Image.BICUBIC)
    lr_img_original = lr_img_model.copy()

    print("ℹ️ Generated LR from HR")

use_metrics = hr_img is not None


# PREPARE INPUT
lr_tensor = transform(lr_img_model).unsqueeze(0).to(device)


# INFERENCE
with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr_tensor = sr_tensor.clamp(0, 1)


# POSTPROCESS
sr_img = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
sr_img = (sr_img * 255).astype("uint8")
sr_img = Image.fromarray(sr_img)


# ENSURE SR = 256×256
target_size = (256, 256)

if sr_img.size != target_size:
    sr_img = sr_img.resize(target_size, Image.BICUBIC)

if hr_img is not None and hr_img.size != target_size:
    hr_img = hr_img.resize(target_size, Image.BICUBIC)


# METRICS
psnr_val, ssim_val = None, None

if use_metrics:
    hr_tensor = transform(hr_img).unsqueeze(0).to(device)

    # ensure tensor size match
    if sr_tensor.shape[-2:] != hr_tensor.shape[-2:]:
        sr_tensor_resized = torch.nn.functional.interpolate(
            sr_tensor, size=hr_tensor.shape[-2:], mode="bicubic", align_corners=False
        )
    else:
        sr_tensor_resized = sr_tensor

    psnr_val = psnr(sr_tensor_resized, hr_tensor)
    ssim_val = ssim(sr_tensor_resized, hr_tensor)

    print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")


# VISUALIZATION
title_map = {
    "edsr": "EDSR",
    "gan": "GAN",
    "unet": "UNet"
}

if use_metrics:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(lr_img_model)
    plt.title("Low Resolution (128×128)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sr_img)
    title = f"{title_map[args.model]} SR (256×256)"
    title += f"\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}"
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr_img)
    plt.title("Ground Truth (256×256)")
    plt.axis("off")

else:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(lr_img_model)
    plt.title("Low Resolution (128×128)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sr_img)
    plt.title(f"{title_map[args.model]} SR (256×256)")
    plt.axis("off")

plt.tight_layout()


# SAVE
model_dir_map = {
    "edsr": "edsr",
    "gan": "gan",
    "unet": "unet"
}

default_names = {
    "edsr": "edsr_output.png",
    "gan": "gan_output.png",
    "unet": "unet_output.png"
}

model_folder = model_dir_map[args.model]
output_name = args.output if args.output != "output.png" else default_names[args.model]

output_path = os.path.join(base_dir, model_folder, output_name)

plt.savefig(output_path, bbox_inches="tight")
print(f"Saved at: {output_path}")

plt.show()
