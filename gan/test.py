import torch
from utils import psnr, ssim


def test_model(
    gen,
    disc,
    test_loader,
    mse,
    bce,
    vgg_loss,
    device,
    adv_weight,
    perc_weight,
    pixel_weight,
    log_file   # 🔥 NEW
):
    gen.eval()

    test_loss = 0
    test_psnr = 0
    test_ssim = 0

    test_adv = 0
    test_perc = 0
    test_pixel = 0

    with torch.no_grad():
        for lr, hr in test_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            fake = gen(lr)
            fake_out = disc(fake)

            adv_loss = adv_weight * bce(fake_out, torch.ones_like(fake_out))
            perc_loss = perc_weight * vgg_loss(fake, hr)
            pixel_loss = pixel_weight * mse(fake, hr)

            loss = adv_loss + perc_loss + pixel_loss

            test_loss += loss.item()
            test_psnr += psnr(fake, hr).item()
            test_ssim += ssim(fake, hr).item()

            test_adv += adv_loss.item()
            test_perc += perc_loss.item()
            test_pixel += pixel_loss.item()

    n = len(test_loader)

    # ================= FORMAT =================
    test_line = (
        f"\nTest → Loss: {test_loss/n:.4f}, "
        f"PSNR: {test_psnr/n:.2f}, "
        f"SSIM: {test_ssim/n:.4f} | "
        f"Adv: {test_adv/n:.4f}, "
        f"Perc: {test_perc/n:.4f}, "
        f"Pixel: {test_pixel/n:.4f}"
    )

    # ================= PRINT =================
    print(test_line)

    # ================= SAVE =================
    with open(log_file, "a") as f:
        f.write(test_line + "\n")