from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import wandb
import time

from vit_pytorch.efficient import ViT
import json
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
# Create a file handler
handler = logging.FileHandler('/tmp/dynamo_grasp_train_log.txt')
logger = logging.getLogger("DYNAMO-GRASP-TRAINING")
logger.addHandler(handler)

logger.info(f"Torch: {torch.__version__}")
logger.info(torch.cuda.is_available())

# Training settings
batch_size = 128
epochs = 500
lr = 1e-4
gamma = 0.7
seed = 42
image_size_h = 256
image_size_w = 256
patch_size = 16
num_classes = 2
channels = 4
dim = 128
depth = 3
heads = 16
mlp_dim = 128
dropout = 0
emb_dropout = 0
json_save = {
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": lr,
    "gamma": gamma,
    "seed": seed,
    "image_size_h": image_size_h,
    "image_size_w": image_size_w,
    "patch_size": patch_size,
    "num_classes": num_classes,
    "channels": channels,
    "dim": dim,
    "depth": depth,
    "heads": heads,
    "mlp_dim": mlp_dim,
    "dropout": dropout,
    "emb_dropout": emb_dropout,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

# augmentation
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class GraspDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        path = self.file_list[idx]

        parts_slashed = path.split("/")
        parts_ = parts_slashed[-1].split("_")

        if parts_[2] == "augment2":
            label_image = np.load(
                parts_slashed[0]
                + "/"
                + parts_slashed[1]
                + "/"
                + parts_slashed[2]
                + "/"
                + parts_slashed[3]
                + "/"
                + parts_[0]
                + "_label_flip_"
                + parts_[-2]
                + "_"
                + parts_[-1]
            )
        else:
            label_image = np.load(
                parts_slashed[0]
                + "/"
                + parts_slashed[1]
                + "/"
                + parts_slashed[2]
                + "/"
                + parts_slashed[3]
                + "/"
                + parts_[0]
                + "_label_"
                + parts_[-2]
                + "_"
                + parts_[-1]
            )

        input_data_path = path
        input_data = np.load(input_data_path)

        trans = transforms.Compose([transforms.ToTensor()])
        input_transformed = trans(input_data)

        return input_transformed, label_image

    def depth_to_point_cloud(self, depth_image):
        cx = depth_image.shape[1] / 2
        cy = depth_image.shape[0] / 2
        fx = 914.0148
        fy = 914.0147
        height, width = depth_image.shape
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        x, y = np.meshgrid(x, y)
        normalized_x = (x - cx) / fx
        normalized_y = (y - cy) / fy
        z = depth_image
        x = normalized_x * z
        y = normalized_y * z
        point_cloud = np.dstack((x, y, z))
        return point_cloud


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT_grasp(nn.Module):
    def __init__(
        self,
        *,
        image_size_h,
        image_size_w,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        grasp_point_dim=2,
        pool="cls",
        channels=4,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        transformer,
    ):
        super().__init__()
        image_height, image_width = pair(image_size_h)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert 1 % 1 == 0, "Frames must be divisible by frame patch size"

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        logger.info(f"{num_patches}, {patch_dim}")

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 32 * 32),
        )

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # Activation and Batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, img):
        img = img.type(torch.float)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        x = self.relu(x)
        x = x.view(-1, 1, 32, 32)

        # Pass through deconvolution layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = torch.sigmoid(x)  # if your image pixels are normalized between 0 and 1
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange("b h i j -> b i j h"),
            nn.LayerNorm(heads),
            Rearrange("b i j h -> b h i j"),
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # attention
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # re-attention
        attn = einsum("b h i j, h g -> b g i j", attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


data_dir = "src/data/Processed_Data"
data_list = glob.glob(os.path.join(data_dir, "*/*_input*.npy"), recursive=True)

logger.info(f"data size: {len(data_list)}")
random.shuffle(data_list)
data_len = len(data_list)
train_list = data_list[: int(data_len)]
train_list, valid_list = train_test_split(train_list, test_size=0.2, random_state=seed)
logger.info(f"Train Data: {len(train_list)}")
logger.info(f"Validation Data: {len(valid_list)}")

train_data = GraspDataset(train_list, transform=train_transforms)
valid_data = GraspDataset(valid_list, transform=test_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)


# effective transformer
efficient_transformer = Linformer(
    dim=dim, seq_len=256 + 1, depth=12, heads=8, k=64  # 15x20 patches + 1 cls-token
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visual transformer
model = ViT_grasp(
    image_size_h=image_size_h,
    image_size_w=image_size_w,
    patch_size=patch_size,
    num_classes=num_classes,
    channels=channels,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout,
    transformer=efficient_transformer,
).to(device)


folder_name = "dynamo_grasp_model"

# # Uncomment wandb code block to use wandb, also uncomment log lines in training loop
# wandb.init(project="my_project", name=folder_name)
# # Configurations (optional)
# config = wandb.config
# config.learning_rate = lr

# loss function
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

torch.cuda.empty_cache()

seconds = time.time()
local_time = time.ctime(seconds)

try:
    os.mkdir("src/models/" + str(folder_name))
except:
    pass

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Number of parameters: {total_params}")

# Y_max are the n highest scored grasp points on an object
Y_max = 15

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device).type(torch.float)
        label = label.unsqueeze(1)

        output = model(data)

        loss_label_masked_all = torch.tensor([], dtype=torch.float).to(device)
        loss_output_masked_all = torch.tensor([], dtype=torch.float).to(device)
        loss_label_masked_all_acc = torch.tensor([], dtype=torch.float).to(device)
        loss_output_masked_all_acc = torch.tensor([], dtype=torch.float).to(device)
        count = 0
        for i in range(label.shape[0]):
            output_single = output[i]
            label_single = label[i]

            mask_single = label_single != -100
            # Apply mask to output and target
            output_masked_single = output_single[mask_single]
            label_masked_single = label_single[mask_single]

            loss_label_masked_all_acc = torch.cat(
                (loss_label_masked_all_acc, label_masked_single)
            )
            loss_output_masked_all_acc = torch.cat(
                (loss_output_masked_all_acc, output_masked_single)
            )

            if label_masked_single.shape[0] < Y_max:
                continue
            count += 1

            loss_label_indices = torch.argsort(output_masked_single, descending=True)

            loss_label_indices = loss_label_indices[:Y_max]

            loss_label_masked_single = label_masked_single[loss_label_indices]
            loss_output_masked_single = output_masked_single[loss_label_indices]

            loss_label_masked_all = torch.cat(
                (loss_label_masked_all, loss_label_masked_single)
            )
            loss_output_masked_all = torch.cat(
                (loss_output_masked_all, loss_output_masked_single)
            )

        if count == 0:
            continue

        loss = criterion(loss_output_masked_all, loss_label_masked_all)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_predictions = torch.abs(
            loss_output_masked_all_acc - loss_label_masked_all_acc
        )
        acc = 1 - correct_predictions.float().mean()

        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        torch.cuda.empty_cache()

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device).type(torch.float)
            label = label.unsqueeze(1)

            val_output = model(data)

            loss_label_masked_all = torch.tensor([], dtype=torch.long).to(device)
            loss_output_masked_all = torch.tensor([], dtype=torch.float).to(device)
            loss_label_masked_all_acc = torch.tensor([], dtype=torch.long).to(device)
            loss_output_masked_all_acc = torch.tensor([], dtype=torch.float).to(device)
            for i in range(label.shape[0]):
                output_single = val_output[i]
                label_single = label[i]

                mask_single = label_single != -100
                # Apply mask to output and target
                output_masked_single = output_single[mask_single]
                label_masked_single = label_single[mask_single]

                loss_label_masked_all_acc = torch.cat(
                    (loss_label_masked_all_acc, label_masked_single)
                )
                loss_output_masked_all_acc = torch.cat(
                    (loss_output_masked_all_acc, output_masked_single)
                )

                loss_label_indices = torch.argsort(
                    output_masked_single, descending=True
                )

                loss_label_indices = loss_label_indices[:Y_max]

                loss_label_masked_single = label_masked_single[loss_label_indices]
                loss_output_masked_single = output_masked_single[loss_label_indices]

                loss_label_masked_all = torch.cat(
                    (loss_label_masked_all, loss_label_masked_single)
                )
                loss_output_masked_all = torch.cat(
                    (loss_output_masked_all, loss_output_masked_single)
                )

            val_loss = criterion(loss_output_masked_all, loss_label_masked_all)
            correct_predictions = torch.abs(
                loss_output_masked_all_acc - loss_label_masked_all_acc
            )
            acc = 1 - correct_predictions.float().mean()

            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        logger.info(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}"
        )
        seconds = time.time()
        local_time = time.ctime(seconds)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            "src/models/"
            + str(folder_name)
            + "/checkpoint_"
            + str(local_time)
            + ".pth",
        )

        save_dir_json = (
            "src/models/"
            + str(folder_name)
            + "/checkpoint_"
            + str(local_time)
            + ".json"
        )
        with open(save_dir_json, "w") as json_file:
            json.dump(json_save, json_file)

    # wandb.log(
    #     {
    #         "epoch": epoch + 1,
    #         "loss": epoch_loss,
    #         "accuracy": epoch_accuracy,
    #         "val_loss": epoch_val_loss,
    #         "val_accuracy": epoch_val_accuracy,
    #     }
    # )
