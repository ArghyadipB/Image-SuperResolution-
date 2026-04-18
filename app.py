import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# =========================
# IMPORT MODELS
# =========================
from unet.model import UNET
from gan.model.generator import Generator
from edsr.model import EDSR
from utils import get_device, psnr, ssim, load_config

# =========================
# DEVICE
# =========================
device = get_device("auto")
print("Using device:", device)

# =========================
# TRANSFORM
# =========================
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# =========================
# LOAD MODELS
# =========================
# UNET
unet_model = UNET().to(device)
unet_model.load_state_dict(torch.load("./unet/saved_models/unet_sr.pth", map_location=device))
unet_model.eval()

# GAN
gan_model = Generator().to(device)
gan_model.load_state_dict(torch.load("./gan/saved_models/generator.pth", map_location=device))
gan_model.eval()

# EDSR
config = load_config("./edsr/config.yaml")

def load_edsr():
    try:
        model = EDSR(
            scale=config["model"]["scale"],
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            res_scale=config["model"]["res_scale"]
        ).to(device)

        model.load_state_dict(torch.load("./edsr/saved_models/edsr.pth", map_location=device))
        print("✅ EDSR loaded with config")
        return model

    except Exception as e:
        print("⚠️ fallback EDSR:", str(e)[:100])

        model = EDSR(scale=2, n_resblocks=16, n_feats=64, res_scale=0.1).to(device)
        model.load_state_dict(torch.load("./edsr/saved_models/edsr.pth", map_location=device))
        return model

edsr_model = load_edsr()
edsr_model.eval()

# =========================
# CORE FUNCTION
# =========================
def run_inference(model_name, lr_img_np, hr_img_np):

    # =========================
    # VALIDATION
    # =========================
    if lr_img_np is None and hr_img_np is None:
        return None, None, "Upload at least one image"

    # convert numpy → PIL
    lr_img = Image.fromarray(lr_img_np.astype("uint8")) if lr_img_np is not None else None
    hr_img = Image.fromarray(hr_img_np.astype("uint8")) if hr_img_np is not None else None

    # =========================
    # FORCE 256x256 INPUT
    # =========================
    if lr_img is not None:
        lr_img = lr_img.resize((256, 256), Image.BICUBIC)

    if hr_img is not None:
        hr_img = hr_img.resize((256, 256), Image.BICUBIC)

    # =========================
    # CREATE 128x128 INPUT
    # =========================
    if lr_img is not None:
        input_128 = lr_img.resize((128, 128), Image.BICUBIC)
    else:
        input_128 = hr_img.resize((128, 128), Image.BICUBIC)

    display_128 = input_128

    # =========================
    # PREPROCESS
    # =========================
    lr_tensor = transform(input_128).unsqueeze(0).to(device)

    # =========================
    # INFERENCE (GPU)
    # =========================
    with torch.no_grad():
        if model_name == "UNet":
            sr_tensor = unet_model(lr_tensor)
        elif model_name == "GAN":
            sr_tensor = gan_model(lr_tensor)
        else:
            sr_tensor = edsr_model(lr_tensor)

    sr_tensor_gpu = sr_tensor  # keep on GPU for metrics

    # =========================
    # METRICS (ONLY IF HR EXISTS)
    # =========================
    psnr_val, ssim_val = None, None

    if hr_img is not None:
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)

        if sr_tensor_gpu.shape[-2:] == hr_tensor.shape[-2:]:
            psnr_val = psnr(sr_tensor_gpu, hr_tensor)
            ssim_val = ssim(sr_tensor_gpu, hr_tensor)

    # =========================
    # MOVE TO CPU FOR DISPLAY
    # =========================
    sr_tensor = sr_tensor.squeeze(0).cpu()

    # =========================
    # CONVERT TO IMAGE
    # =========================
    sr_img = (sr_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    sr_img = Image.fromarray(sr_img)

    # ensure SR output is 256x256
    sr_img = sr_img.resize((256, 256), Image.BICUBIC)

    # =========================
    # FORMAT OUTPUT
    # =========================
    display_128 = np.array(display_128)
    sr_img = np.array(sr_img)

    if psnr_val is not None:
        metrics_text = f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}"
    else:
        metrics_text = "No HR provided → metrics not computed"

    return display_128, sr_img, metrics_text


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:

    gr.Markdown("# 🔬 Super Resolution App")

    model_choice = gr.Dropdown(
        ["UNet", "GAN", "EDSR"],
        value="EDSR",
        label="Select Model"
    )

    with gr.Row():
        lr_input = gr.Image(label="Input LR (256x256)")
        hr_input = gr.Image(label="Input HR (256x256, Optional)")

    run_btn = gr.Button("Run")

    with gr.Row():
        lr_display = gr.Image(label="128x128 Downsampled Input")
        sr_output = gr.Image(label="256x256 Super Resolution Output")

    metrics_output = gr.Textbox(label="Metrics")

    run_btn.click(
        fn=run_inference,
        inputs=[model_choice, lr_input, hr_input],
        outputs=[lr_display, sr_output, metrics_output]
    )

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    demo.launch()