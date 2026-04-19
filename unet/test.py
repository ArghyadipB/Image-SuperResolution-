"""
Model Evaluation (Testing) for Super-Resolution

This module evaluates a trained super-resolution model on a test dataset.
It computes quantitative metrics and logs the performance for analysis.
"""
import torch
from utils import psnr, ssim
import matplotlib.pyplot as plt

# TEST FUNCTION
def test_model(model, test_loader, mse, vgg_loss, perc_weight, device, log_file):
    model.eval()
    test_loss = test_psnr = test_ssim = 0

    with torch.no_grad():
        for lr_img, hr_img in test_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr = model(lr_img)

            loss = mse(sr, hr_img) + perc_weight * vgg_loss(sr, hr_img)

            test_loss += loss.item()
            test_psnr += psnr(sr, hr_img).item()
            test_ssim += ssim(sr, hr_img).item()

    n_test = len(test_loader)

    log_line = (
    f"\nTest → Loss: {test_loss/n_test:.4f}, "
    f"PSNR: {test_psnr/n_test:.2f}, "
    f"SSIM: {test_ssim/n_test:.4f}"
               )

    print(log_line)

    with open(log_file, "a") as f:
        f.write(log_line + "\n")
