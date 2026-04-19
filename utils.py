"""
Utility Functions for Super-Resolution Pipeline

This module provides helper functions used across training and inference
for super-resolution models such as EDSR, GAN, and UNet. It includes
configuration loading, device management, model saving, and evaluation metrics.
"""
import os
import yaml
import torch
import torch.nn.functional as F

# CONFIG LOAD
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# DEVICE
def get_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available")
        return "cuda"
    elif device_str == "cpu":
        return "cpu"
    else:
        raise ValueError("Invalid device")

# MODEL SAVE
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


# METRICS
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = pred.mean()
    mu_y = target.mean()

    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))