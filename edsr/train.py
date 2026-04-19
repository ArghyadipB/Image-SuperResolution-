"""
EDSR Training Loop with Pixel, Perceptual, and Edge Losses

This module implements the training and validation pipeline for an EDSR-based
super-resolution model. It optimizes image reconstruction quality using a
combination of pixel-level, perceptual, and edge-aware losses.
"""
from utils import psnr, ssim
import torch

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    pixel_loss_fn,
    perceptual_loss_fn,
    edge_loss_fn,
    device,
    epochs,
    lambda_pixel,
    lambda_perc,
    lambda_edge,
    log_file
):
    for epoch in range(epochs):

        model.train()
        train_loss = train_psnr = train_ssim = 0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            pixel = pixel_loss_fn(sr, hr)
            perc = perceptual_loss_fn(sr, hr)
            edge = edge_loss_fn(sr, hr)

            loss = lambda_pixel * pixel + \
                   lambda_perc * perc + \
                   lambda_edge * edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += psnr(sr, hr).item()
            train_ssim += ssim(sr, hr).item()

        # VALIDATION
        model.eval()
        val_loss = val_psnr = val_ssim = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)

                sr = model(lr)

                loss = pixel_loss_fn(sr, hr)
                val_loss += loss.item()
                val_psnr += psnr(sr, hr).item()
                val_ssim += ssim(sr, hr).item()

        n_train = len(train_loader)
        n_val = len(val_loader)

        train_line = (
            f"Epoch {epoch+1} | Train Loss: {train_loss/n_train:.4f}, "
            f"PSNR: {train_psnr/n_train:.2f}, SSIM: {train_ssim/n_train:.4f}"
        )

        val_line = (
            f"Epoch {epoch+1} | Val Loss: {val_loss/n_val:.4f}, "
            f"PSNR: {val_psnr/n_val:.2f}, SSIM: {val_ssim/n_val:.4f}"
        )

        print(train_line)
        print(val_line)

        with open(log_file, "a") as f:
            f.write(train_line + "\n")
            f.write(val_line + "\n")