from __future__ import print_function

import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from linformer import Linformer
from torch.utils.data import Dataset
from torchvision import transforms
import open3d as o3d
import cv2

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logging
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
# Create a file handler
handler = logging.FileHandler("/tmp/dynamo_grasp_inference_log.txt")
logger = logging.getLogger("DYNAMO-GRASP-INFERENCE")
logger.addHandler(handler)

logger.info(f"Torch: {torch.__version__}")
logger.info(torch.cuda.is_available())

# Training settings
batch_size = 128
epochs = 500
lr = 5e-4
gamma = 0.7
seed = 42
image_size_h = 256
image_size_w = 256
patch_size = 16
num_classes = 2
channels = 4
dim = 128  # 1024
depth = 3
heads = 16
mlp_dim = 128
dropout = 0
emb_dropout = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)
device = "cuda:0"

# augmentation
train_transforms = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


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


class GraspDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength


class inference:
    def __init__(self):
        efficient_transformer = Linformer(
            dim=dim, seq_len=256 + 1, depth=12, heads=8, k=64
        )

        self.model = ViT_grasp(
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
        self.bin_crop_dim = 256
        self.bin_back = 1.225
        self.background_value = 1.071
        self.crop_coords = [156, 346, 236, 431]

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

    def estimate_normals(
        self,
        point_cloud,
        obj_mask,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=624),
    ):
        valid_idx = np.zeros_like(obj_mask, dtype=np.bool)
        coord1, coord2 = np.nonzero(obj_mask)
        coord1_min, coord1_max = coord1.min(), coord1.max()
        coord2_min, coord2_max = coord2.min(), coord2.max()
        valid_idx[coord1_min : coord1_max + 1, coord2_min : coord2_max + 1] = 1
        valid_idx = valid_idx & (point_cloud[..., 2] != 0)
        height, width, _ = point_cloud.shape
        point_cloud_valid = point_cloud[valid_idx]
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_valid))
        pc_o3d.estimate_normals(search_param)
        pc_o3d.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
        pc_o3d.normalize_normals()
        normals = np.array(pc_o3d.normals).astype(np.float32)
        normal_map = np.zeros([height, width, 3], dtype=np.float32)
        normal_map[valid_idx] = normals
        return normal_map

    def normal_to_angles(self, normals):
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        dot_x = np.dot(normals, x_axis)
        dot_y = np.dot(normals, y_axis)
        dot_z = np.dot(normals, z_axis)
        angles_x = np.arccos(dot_x) * 180 / np.pi
        angles_y = np.arccos(dot_y) * 180 / np.pi
        angles_z = np.arccos(dot_z) * 180 / np.pi
        return angles_x, angles_y, angles_z

    def run_model(self, depth_image, segmask, id):
        self.model.load_state_dict(
            torch.load(
                "inference/model/dynamo_grasp_model.pth",
                map_location=device,
            )["model_state_dict"]
        )
        self.model.to(device)
        self.model.eval()

        segmask_numpy = np.zeros_like(segmask)
        segmask_numpy[segmask == id] = 1

        depth_mask = np.zeros_like(segmask)
        depth_mask[segmask != 0] = 1

        y, x = np.where(segmask_numpy == 1)

        center_x, center_y = np.mean(x), np.mean(y)

        top_left_x = int(center_x) - int(self.bin_crop_dim / 2)
        top_left_y = int(center_y) - int(self.bin_crop_dim / 2)
        bottom_right_x = int(center_x) + int(self.bin_crop_dim / 2)
        bottom_right_y = int(center_y) + int(self.bin_crop_dim / 2)

        centroid = np.array([top_left_x, top_left_y])

        depth_image[depth_mask != 1] = self.bin_back
        depth_image[: self.crop_coords[0], :] = self.background_value
        depth_image[:, : self.crop_coords[2]] = self.background_value
        depth_image[self.crop_coords[1] :, :] = self.background_value
        depth_image[:, self.crop_coords[3] :] = self.background_value

        top_left_x_pad = int(center_x) - int(self.bin_crop_dim / 2)
        top_left_y_pad = int(center_y) - int(self.bin_crop_dim / 2)
        bottom_right_x_pad = int(center_x) + int(self.bin_crop_dim / 2)
        bottom_right_y_pad = int(center_y) + int(self.bin_crop_dim / 2)

        depth_processed = depth_image[
            top_left_y_pad:bottom_right_y_pad, top_left_x_pad:bottom_right_x_pad
        ]

        segmask_processed = segmask_numpy[
            top_left_y_pad:bottom_right_y_pad, top_left_x_pad:bottom_right_x_pad
        ]

        point_cloud = self.depth_to_point_cloud(depth_processed)

        segmask_processed = np.expand_dims(segmask_processed, axis=-1)
        input_data = np.concatenate((point_cloud, segmask_processed), axis=-1)
        input_data = input_data.astype(np.double)

        trans = transforms.Compose([transforms.ToTensor()])
        input_transformed = trans(input_data)

        input_transformed = input_transformed.unsqueeze(0).type(torch.float).to(device)

        output = self.model(input_transformed)
        output = torch.squeeze(output)

        output = output.unsqueeze(2)
        output = output.cpu().detach().numpy() * segmask_processed

        max_value = np.amax(output)
        grasp_point = None
        if max_value < 0.7:
            max_coordinates = np.argwhere(output == max_value)
            grasp_point = np.array([max_coordinates[0][1], max_coordinates[0][0]])
        else:
            try:
                points = np.column_stack(np.where(output >= 0.9))
                # Perform DBSCAN on the points
                db = DBSCAN(eps=1.5, min_samples=5).fit(
                    points
                )  # You may need to adjust the parameters
                # Find the labels of the clusters that each point belongs to
                labels = db.labels_
                # Ignore noises in the cluster computation (noises are denoted by -1)
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                score_points = output[output >= 0.9]

                # Calculate the mean score for each cluster and identify the
                # cluster with the highest mean score
                max_avg_score = 0
                max_cluster_size = 0

                normalize_size = np.array([])
                for cluster_id in np.unique(labels):
                    if cluster_id == -1:
                        continue
                    normalize_size = np.append(
                        normalize_size, len(points[labels == cluster_id])
                    )
                normalize_size = normalize_size / np.max(normalize_size) * 0.025

                count = 0
                # Calculate the mean coordinate for each cluster
                for cluster_id in np.unique(labels):
                    if cluster_id == -1:
                        continue  # Skip noise

                    cluster_points = points[labels == cluster_id]
                    cluster_scores = score_points[labels == cluster_id]

                    mean_coordinate = cluster_points.mean(axis=0)
                    avg_score = cluster_scores.mean()
                    avg_cluster_size_score = (
                        cluster_scores.mean() + normalize_size[count]
                    )
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                        grasp_point = np.array(
                            [mean_coordinate[1], mean_coordinate[0]]
                        ).astype(np.int16)
                    # if avg_cluster_size_score > max_cluster_size:
                    #     max_cluster_size = avg_cluster_size_score
                    #     grasp_point = np.array(
                    #         [mean_coordinate[1], mean_coordinate[0]]
                    #     ).astype(np.int16)
                    count += 1
            except:
                pass
        # If all above methods fail to get a grasp point, then use the following method
        try:
            if grasp_point == None:
                max_coordinates = np.argwhere(output == max_value)
                avg_row = 0.0
                avg_col = 0.0
                for i in range(len(max_coordinates)):
                    max_row, max_col = max_coordinates[i][1], max_coordinates[i][0]
                    avg_row += max_row
                    avg_col += max_col

                avg_row /= len(max_coordinates)
                avg_col /= len(max_coordinates)

                min_dist = sys.maxsize
                grasp_point = None
                for i in range(len(max_coordinates)):
                    max_row, max_col = max_coordinates[i][1], max_coordinates[i][0]
                    temp_first_point = np.array([max_row, max_col])
                    temp_second_point = np.array([avg_row, avg_col])
                    dist = np.linalg.norm(temp_second_point - temp_first_point)
                    if dist < min_dist:
                        dist = min_dist
                        grasp_point = temp_second_point.astype(np.int16)
        except:
            pass
        return grasp_point, centroid, output


if __name__ == "__main__":
    policy_inference = inference()
    depth_image = np.load("inference/test_data/depth_image.npy")
    segmask_image = np.load("inference/test_data/segmask.npy")
    rgb_image = cv2.imread("inference/test_data/rgb.png", cv2.IMREAD_COLOR)
    print(f"Unique mask id, {np.unique(segmask_image)}")
    # Select mask id from segmask_image(don't select 0, as it is a background)
    grasp_point, centroid, output = policy_inference.run_model(
        depth_image, segmask_image, 49
    )
    grasp_point = grasp_point + centroid
    cv2.circle(rgb_image, (grasp_point[0], grasp_point[1]), 3, (0, 255, 0), -1)
    cv2.circle(rgb_image, (grasp_point[0], grasp_point[1]), 5, (0, 255, 0), 1)
    cv2.imshow("RGB", rgb_image)
    cv2.waitKey(0)
