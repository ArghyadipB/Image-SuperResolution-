"""
Super-Resolution Dataset Loader

This dataset class is designed for training and evaluating super-resolution
models such as EDSR, GAN-based models, and UNet. It supports two different
data-loading strategies depending on the model type.
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


# DATASET CLASS
class SRDataset(Dataset):
    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        scale=1
    ):
        self.hr_path = os.path.join(root, mode, "high_res")
        self.lr_path = os.path.join(root, mode, "low_res")

        self.hr_imgs = sorted(os.listdir(self.hr_path))
        self.lr_imgs = sorted(os.listdir(self.lr_path))

        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, idx):
        hr = Image.open(
            os.path.join(self.hr_path, self.hr_imgs[idx])
        ).convert("RGB")
        if self.scale > 1:
            lr = hr.resize(
                (hr.width // self.scale, hr.height // self.scale),
                Image.BICUBIC
            )
        else:
            lr = Image.open(
                os.path.join(self.lr_path, self.lr_imgs[idx])
            ).convert("RGB")

        if self.transform:
            hr = self.transform(hr)
            lr = self.transform(lr)

        return lr, hr