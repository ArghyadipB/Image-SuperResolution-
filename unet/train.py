"""
Training Loop for Super-Resolution Models

This module implements the training and validation pipeline for
super-resolution models such as UNet, GAN, or EDSR. It optimizes the model
using a combination of pixel-wise and perceptual losses, while tracking
quantitative metrics during training.
"""
import torch
from utils import psnr, ssim

def train_model(model, train_loader, val_loader, optimizer, mse, vgg_loss, perc_weight, device, epochs, log_file):

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = train_psnr = train_ssim = 0

        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr = model(lr_img)

            loss = mse(sr, hr_img) + perc_weight * vgg_loss(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += psnr(sr, hr_img).item()
            train_ssim += ssim(sr, hr_img).item()

        # VALIDATION
        model.eval()
        val_loss = val_psnr = val_ssim = 0

        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                sr = model(lr_img)

                loss = mse(sr, hr_img) + perc_weight * vgg_loss(sr, hr_img)

                val_loss += loss.item()
                val_psnr += psnr(sr, hr_img).item()
                val_ssim += ssim(sr, hr_img).item()

        n_train = len(train_loader)
        n_val = len(val_loader)

        log_line = (
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss/n_train:.4f}, PSNR: {train_psnr/n_train:.2f}, SSIM: {train_ssim/n_train:.4f} | "
                f"Val Loss: {val_loss/n_val:.4f}, PSNR: {val_psnr/n_val:.2f}, SSIM: {val_ssim/n_val:.4f}"
                   )

        print(log_line)

        with open(log_file, "a") as f:
            f.write(log_line + "\n")