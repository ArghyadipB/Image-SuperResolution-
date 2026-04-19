# Image Super Resolution Project

This repository contains the implementation of an **Image Super Resolution (SR)** project completed as part of the coursework at **IIT Kanpur** for the subject *Computer Vision and Deep Learning*.

---

## 📌 Overview

Image Super Resolution (SR) aims to reconstruct a high-resolution (HR) image from its low-resolution (LR) counterpart. This project explores and compares multiple deep learning approaches for SR, focusing on reconstruction quality, high-frequency detail recovery, and perceptual performance.

The implementation includes:

* Baseline SR models
* Enhanced architectures
* Custom loss functions (including a novel triplet-based formulation)
* Evaluation pipelines and visualization tools

---

## 🚀 Key Features

* 🔹 Implementation of multiple SR models:

  * EDSR (Enhanced Deep Super Resolution)
  * GAN-based Super Resolution
  * UNet-based architecture

* 🔹 Custom training pipeline with:

  * Bicubic downsampling for self-supervision
  * Flexible HR/LR input handling

* 🔹 Novel contribution:

  * Integration of **Triplet Loss** for improved feature discrimination and sharper reconstruction

* 🔹 Evaluation metrics:

  * PSNR (Peak Signal-to-Noise Ratio)
  * Visual comparison outputs

* 🔹 Modular and extensible code structure

---

## 🧠 Models Implemented

### 1. EDSR

* Residual learning-based architecture
* PixelShuffle upsampling
* Strong baseline for SR tasks

### 2. GAN-based SR

* Generator-discriminator setup
* Improves perceptual quality
* Produces sharper textures

### 3. UNet-based SR

* Encoder-decoder with skip connections
* Captures both local and global features

---

## 🧪 Methodology

### Data Processing

* HR images are resized to generate LR inputs using **bicubic interpolation**
* Supports:

  * HR-only input (self-supervised LR generation)
  * LR-only input (downsampling before model inference)

### Training Strategy

* Input: 128×128 LR patches
* Target: 256×256 HR images
* Loss functions:

  * Pixel loss (L1)
  * Novel Triplet Loss (proposed)

---

## 📊 Results

* Comparison across models using:

  * Quantitative metrics (PSNR)
  * Qualitative outputs (image comparisons)

* Observations:

  * EDSR performs well on reconstruction fidelity
  * GAN improves perceptual sharpness
  * Triplet loss enhances high-frequency detail recovery

##

---

## ⚙️ Installation

```bash
git clone https://github.com/ArghyadipB/Image-SuperResolution-.git
cd Image-SuperResolution-
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Usage

### Inference:

Our inference works in 3 modes even though one of the modes is actual representative of the synthetic self-supervised training(mode 2).

Mode 1: In this mode, only a low-resolution image of size 256×256 is provided. The image is first bicubicly downsampled to 128×128 and then passed to the model as input. Since no ground truth high-resolution image is available, no evaluation metrics such as PSNR or SSIM are computed.

Example: python infer.py --model edsr --lr <lr_image_path>

Mode 2: In this mode, only a high-resolution image of size 256×256 is provided. To create the model input, the HR image is bicubicly downsampled to 128×128. This generated low-resolution image is then passed through the model to produce a 256×256 super-resolved output. Since the original HR image is available, PSNR and SSIM are computed between the predicted SR image and the ground truth HR image. 

Example: python infer.py --model edsr --hr <hr_image_path>

Mode 3: In this mode, both LR and HR images of size 256×256 are provided. The LR image is bicubicly downsampled to 128×128 and used as input to the model. The model generates a 256×256 super-resolved image, which is then compared with the ground truth HR image to compute PSNR and SSIM.

```bash
Example: python infer.py --model edsr --lr <lr_image_path> --hr <hr_image_path>

Note: There is all a gradio interface which does the same thing but with proper
visualization.
```

---

## 🧩 Custom Contributions

* ✔️ Integration of **Triplet Loss** into SR framework
* ✔️ Flexible pipeline for handling HR-only or LR-only inputs
* ✔️ Comparative evaluation across multiple architectures

---

## 📈 Future Work

* Incorporate perceptual loss (VGG-based)
* Extend to real-world degradation (not just bicubic)
* Deploy as a web application (Gradio interface)
* Explore transformer-based SR models

---

## 👨‍💻 Authors

* Project developed as part of coursework at **IIT Kanpur**

---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

* Inspired by original EDSR and SRGAN papers
* Course instructors and peers for guidance and feedback

---
