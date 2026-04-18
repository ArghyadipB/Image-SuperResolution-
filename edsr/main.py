import sys
import os

# allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2

# shared modules
from dataset import SRDataset
from utils import load_config, get_device, save_model
from visualize import visualize_results

# edsr modules
from edsr.model import EDSR
from edsr.loss import VGGPerceptualLoss, edge_loss
from edsr.train import train_model
from edsr.test import test_model


# =========================
# CONFIG
# =========================
base_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

config_path = os.path.join(base_dir, "config.yaml")
config = load_config(config_path)

device = get_device(config["system"]["device"])
print("Using device:", device)

torch.manual_seed(config["system"]["seed"])

scale = config["model"]["scale"]


# =========================
# DATA
# =========================
dataset_root = os.path.join(root_dir, config["dataset"]["root"])

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset = SRDataset(
    dataset_root,
    "train",
    transform,
    scale=scale
)

val_dataset_full = SRDataset(
    dataset_root,
    "val",
    transform,
    scale=scale
)


# =========================
#  FIXED SPLIT (SHARED)
# =========================
split_path = os.path.join(root_dir, "split_indices.pt")

if not os.path.exists(split_path):
    print("Creating new data split...")

    val_size = int(config["dataset"]["val_split"] * len(val_dataset_full))
    test_size = len(val_dataset_full) - val_size

    generator = torch.Generator().manual_seed(config["system"]["seed"])

    val_dataset, test_dataset = random_split(
        val_dataset_full,
        [val_size, test_size],
        generator=generator
    )

    torch.save({
        "val_idx": val_dataset.indices,
        "test_idx": test_dataset.indices
    }, split_path)

else:
    print("Loading existing split...")

    split = torch.load(split_path)

    val_dataset = Subset(val_dataset_full, split["val_idx"])
    test_dataset = Subset(val_dataset_full, split["test_idx"])

print(f"Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")


# =========================
# LOADERS
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=config["dataset"]["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["dataset"]["batch_size"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["dataset"]["batch_size"],
    shuffle=False
)


# =========================
# DEBUG (RUN ONCE)
# =========================
lr, hr = next(iter(train_loader))
print("LR shape:", lr.shape)
print("HR shape:", hr.shape)


# =========================
# MODEL
# =========================
model = EDSR(
            scale=scale,
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            res_scale=config["model"]["res_scale"]
        ).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=config["training"]["lr"]
)

pixel_loss = nn.L1Loss()
perc_loss = VGGPerceptualLoss().to(device)

lambda_pixel = config["loss"]["lambda_pixel"]
lambda_perc = config["loss"]["lambda_perc"]
lambda_edge = config["loss"]["lambda_edge"]


# =========================
# LOG FILE
# =========================
log_file = os.path.join(base_dir, "results.txt")
open(log_file, "w").close()


# =========================
# TRAIN
# =========================
train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    pixel_loss,
    perc_loss,
    edge_loss,
    device,
    config["training"]["epochs"],
    lambda_pixel,
    lambda_perc,
    lambda_edge,
    log_file
)


# =========================
# TEST
# =========================
test_model(
    model,
    test_loader,
    pixel_loss,
    device,
    log_file
)


# =========================
# VISUALIZE
# =========================
visualize_results(model, test_loader, device)


# =========================
# SAVE MODEL
# =========================
save_path = os.path.join(base_dir, config["save"]["path"])
save_model(model, save_path)

print("Model saved!")