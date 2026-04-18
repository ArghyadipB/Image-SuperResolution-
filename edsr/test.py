from utils import psnr, ssim
import torch

def test_model(model, loader, loss_fn, device, log_file):
    model.eval()

    total_loss = total_psnr = total_ssim = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            loss = loss_fn(sr, hr)

            total_loss += loss.item()
            total_psnr += psnr(sr, hr).item()
            total_ssim += ssim(sr, hr).item()

    n = len(loader)

    line = (
        f"\nTest → Loss: {total_loss/n:.4f}, "
        f"PSNR: {total_psnr/n:.2f}, "
        f"SSIM: {total_ssim/n:.4f}"
    )

    print(line)

    with open(log_file, "a") as f:
        f.write(line + "\n")