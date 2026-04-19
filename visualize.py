"""
Visualization Utility for Super-Resolution Results

This module provides a helper function to visually compare the performance
of super-resolution models such as EDSR, GAN, and UNet. It displays
Low-Resolution (LR), Super-Resolved (SR), and High-Resolution (HR) images
side by side for qualitative evaluation.
"""


import torch
import matplotlib.pyplot as plt

# VISUALIZE FUNCTION
def visualize_results(model, loader, device, num_images=3):
    model.eval()

    lr, hr = next(iter(loader))

    with torch.no_grad():
        sr = model(lr.to(device)).cpu()

    plt.figure(figsize=(12, 8))

    for i in range(num_images):
        # Low
        plt.subplot(3, num_images, i + 1)
        plt.imshow(lr[i].permute(1, 2, 0))
        plt.title("Low")
        plt.axis("off")

        # SR
        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(sr[i].permute(1, 2, 0))
        plt.title("SR")
        plt.axis("off")

        # High
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(hr[i].permute(1, 2, 0))
        plt.title("High")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
