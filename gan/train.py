"""
GAN-Based Super-Resolution Training Loop

This module implements the training and validation pipeline for a GAN-based
super-resolution model. It jointly trains a Generator and Discriminator using
a combination of adversarial, perceptual, and pixel-wise losses.
"""

import torch
from utils import psnr, ssim


def train_model(
    gen,
    disc,
    train_loader,
    val_loader,
    opt_g,
    opt_d,
    mse,
    bce,
    vgg_loss,
    device,
    epochs,
    adv_weight,
    perc_weight,
    pixel_weight,
    log_file
):
    for epoch in range(epochs):

        # TRAIN
        gen.train()
        disc.train()

        train_loss = 0
        train_psnr = 0
        train_ssim = 0

        train_adv = 0
        train_perc = 0
        train_pixel = 0

        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            fake = gen(lr)

            # DISCRIMINATOR
            real_out = disc(hr)
            fake_out = disc(fake.detach())

            loss_d = bce(real_out, torch.ones_like(real_out)) + \
                     bce(fake_out, torch.zeros_like(fake_out))

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # GENERATOR
            fake_out = disc(fake)

            adv_loss = adv_weight * bce(fake_out, torch.ones_like(fake_out))
            perc_loss = perc_weight * vgg_loss(fake, hr)
            pixel_loss = pixel_weight * mse(fake, hr)

            loss_g = adv_loss + perc_loss + pixel_loss

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # METRICS
            train_loss += loss_g.item()
            train_psnr += psnr(fake, hr).item()
            train_ssim += ssim(fake, hr).item()

            train_adv += adv_loss.item()
            train_perc += perc_loss.item()
            train_pixel += pixel_loss.item()

        # VALIDATION
        gen.eval()

        val_loss = 0
        val_psnr = 0
        val_ssim = 0

        val_adv = 0
        val_perc = 0
        val_pixel = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)

                fake = gen(lr)
                fake_out = disc(fake)

                adv_loss = adv_weight * bce(fake_out, torch.ones_like(fake_out))
                perc_loss = perc_weight * vgg_loss(fake, hr)
                pixel_loss = pixel_weight * mse(fake, hr)

                loss = adv_loss + perc_loss + pixel_loss

                val_loss += loss.item()
                val_psnr += psnr(fake, hr).item()
                val_ssim += ssim(fake, hr).item()

                val_adv += adv_loss.item()
                val_perc += perc_loss.item()
                val_pixel += pixel_loss.item()

        # FORMAT LOGS
        n_train = len(train_loader)
        n_val = len(val_loader)

        train_line = (
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss/n_train:.4f}, "
            f"PSNR: {train_psnr/n_train:.2f}, "
            f"SSIM: {train_ssim/n_train:.4f} | "
            f"Adv: {train_adv/n_train:.4f}, "
            f"Perc: {train_perc/n_train:.4f}, "
            f"Pixel: {train_pixel/n_train:.4f}"
        )

        val_line = (
            f"Epoch {epoch+1} | "
            f"Val Loss: {val_loss/n_val:.4f}, "
            f"PSNR: {val_psnr/n_val:.2f}, "
            f"SSIM: {val_ssim/n_val:.4f} | "
            f"Adv: {val_adv/n_val:.4f}, "
            f"Perc: {val_perc/n_val:.4f}, "
            f"Pixel: {val_pixel/n_val:.4f}"
        )

        # PRINT
        print(train_line)
        print(val_line)

        # SAVE TO FILE
        with open(log_file, "a") as f:
            f.write(train_line + "\n")
            f.write(val_line + "\n")