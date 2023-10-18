import json
import numpy as np
import os
import glob
import shutil
import cv2
import matplotlib.pyplot as plt
import random

bin_crop_dim = 256


def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = point_cloud[:, :, 0].flatten()
    y = point_cloud[:, :, 1].flatten()
    z = point_cloud[:, :, 2].flatten()
    ax.scatter(x, y, z, s=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def depth_to_point_cloud(depth_image):
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


def flip_segmask_depth(depth_image, segmask):
    depth_flip = cv2.flip(depth_image, 1)
    segmask_flip = cv2.flip(segmask, 1)
    point_cloud_flipped = depth_to_point_cloud(depth_flip)
    segmask_flip = np.expand_dims(segmask_flip, axis=-1)
    input_data_augment = np.concatenate((point_cloud_flipped, segmask_flip), axis=-1)
    input_data_augment = input_data_augment.astype(np.double)
    return input_data_augment


def add_noise_to_xyz(xyz_img, depth_img):
    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    xyz_img = xyz_img.copy()
    H, W, C = xyz_img.shape
    gp_rescale_factor = np.random.randint(12, 20)
    gp_scale = np.random.uniform(0, 0.003)
    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(
        loc=0.0, scale=gp_scale, size=(small_H, small_W, C)
    )
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]
    return xyz_img


id = -1
# Specify the path to the simulated data directory here.
# You can add multiple paths to the list.
PATH_list = ["test_grasp_data/"]
output_dir = "src/data/Processed_Data/"
os.makedirs(output_dir, exist_ok=True)
raw_data_dir = "../scenario_grasp_configurations/"

bin_back = 1.225
background_value = 1.071
# Bin crop coordinates
crop_coords = [156, 346, 236, 431]

for path in PATH_list:
    data_dir = os.path.join(raw_data_dir, path)

    id += 1
    data_list = glob.glob(data_dir + "/**/depth*.npy", recursive=True)
    for i in range(len(data_list)):
        path = data_list[i]

        parts_slashed = path.split("/")
        parts_ = parts_slashed[-1].split("_")
        parts_dot = parts_[-1].split(".")

        json_list = glob.glob(
            data_dir
            + str(parts_slashed[-2])
            + "/json_data_"
            + str(parts_slashed[-2])
            + "_*_"
            + str(parts_dot[0])
            + ".json",
            recursive=True,
        )
        if len(json_list) == 0:
            continue
        segmask_path = (
            data_dir
            + str(parts_slashed[-2])
            + "/segmask_"
            + parts_[-2]
            + "_"
            + parts_[-1]
        )
        depth_path = path

        segmask = np.load(segmask_path)
        depth_image = np.load(depth_path)

        with open(json_list[0]) as json_file:
            data_id = json.load(json_file)

        segmask_numpy = np.zeros_like(segmask)
        target_id = data_id["object_id"]
        segmask_numpy[segmask == target_id] = 1

        depth_mask = np.zeros_like(segmask)
        depth_mask[segmask != 0] = 1

        y, x = np.where(segmask_numpy == 1)
        center_x, center_y = np.mean(x), np.mean(y)

        top_left_x = int(center_x) - int(bin_crop_dim / 2)
        top_left_y = int(center_y) - int(bin_crop_dim / 2)
        bottom_right_x = int(center_x) + int(bin_crop_dim / 2)
        bottom_right_y = int(center_y) + int(bin_crop_dim / 2)

        # Update depth image to simulate a centered bin around the object with a uniform background.
        depth_image[depth_mask != 1] = bin_back

        depth_image[: crop_coords[0], :] = background_value
        depth_image[:, : crop_coords[2]] = background_value
        depth_image[crop_coords[1] :, :] = background_value
        depth_image[:, crop_coords[3] :] = background_value
        
        top_left_x_pad = int(center_x) - int(bin_crop_dim / 2)
        top_left_y_pad = int(center_y) - int(bin_crop_dim / 2)
        bottom_right_x_pad = (
            int(center_x) + int(bin_crop_dim / 2)
        )
        bottom_right_y_pad = (
            int(center_y) + int(bin_crop_dim / 2)
        )
        
        depth_processed = depth_image[
            top_left_y_pad:bottom_right_y_pad, top_left_x_pad:bottom_right_x_pad
        ]

        segmask_processed = segmask_numpy[
            top_left_y_pad:bottom_right_y_pad, top_left_x_pad:bottom_right_x_pad
        ]
        
        if depth_processed.shape[0] != int(bin_crop_dim) or depth_processed.shape[
            1
        ] != int(bin_crop_dim):
            continue

        point_cloud = depth_to_point_cloud(depth_processed)
        # visualize_point_cloud(point_cloud)

        segmask_processed = np.expand_dims(segmask_processed, axis=-1)
        input_data = np.concatenate((point_cloud, segmask_processed), axis=-1)
        input_data = input_data.astype(np.double)

        # ellipse noise xyz
        augment_point_cloud = add_noise_to_xyz(point_cloud, depth_processed)
        input_data_augment = np.concatenate(
            (augment_point_cloud, segmask_processed), axis=-1
        )
        input_data_augment = input_data_augment.astype(np.double)

        # flip depth image and segmask
        input_data_augment_flip = flip_segmask_depth(depth_processed, segmask_processed)

        os.makedirs(output_dir + parts_[-2], exist_ok=True)
        np.save(
            output_dir
            + parts_[-2]
            + "/"
            + str(id)
            + "_input_data_"
            + parts_[-2]
            + "_"
            + str(parts_dot[0])
            + ".npy",
            input_data,
        )
        np.save(
            output_dir
            + parts_[-2]
            + "/"
            + str(id)
            + "_input_augment1_"
            + parts_[-2]
            + "_"
            + str(parts_dot[0])
            + ".npy",
            input_data_augment,
        )

        label_image = np.ones((int(bin_crop_dim), int(bin_crop_dim))) * (-100)
        label_image = label_image.astype(np.float64)

        for json_files in json_list:
            with open(json_files) as json_file:
                grasp_data = json.load(json_file)
            grasp_point = np.array(grasp_data["grasp point"])
            label = int(grasp_data["success"])
            if int(grasp_data["oscillation"]):
                label = 0
            elif int(grasp_data["penetration"]):
                label = 1
            if int(grasp_data["unreachable"]):
                label = 0

            grasp_point[0] = int(bin_crop_dim / 2) + (grasp_point[0] - center_x)
            grasp_point[1] = int(bin_crop_dim / 2) + (grasp_point[1] - center_y)

            poses = grasp_data["object_disp"]
            disp = 0.0
            temp_quat = 0.0
            if len(poses) != 0:
                pose1 = poses[0]
                for i in range(0, int(len(grasp_data["object_disp"])), 5):
                    pose2 = poses[i]
                    translation1, quaternion1 = np.split(pose1, [3])
                    translation2, quaternion2 = np.split(pose2, [3])
                    translation_norm = np.linalg.norm(translation1 - translation2)
                    quaternion_norm = 1 - abs(np.dot(quaternion1, quaternion2))
                    disp += translation_norm + quaternion_norm
                    pose1 = pose2

            label = label - disp
            if label <= 0.2:
                label = 0.0
            elif label < 0.7:
                label = 0.7

            label_image[int(grasp_point[1])][int(grasp_point[0])] = label

            slash_parts = json_files.split("/")
            parts_combined = json_files.split("_")
            parts = parts_combined[-1].split(".")

        label_image_flip = cv2.flip(label_image, 1)

        np.save(
            output_dir
            + parts_[-2]
            + "/"
            + str(id)
            + "_label_"
            + parts_[-2]
            + "_"
            + str(parts_dot[0])
            + ".npy",
            label_image,
        )
        if random.random() < 0.5:
            np.save(
                output_dir
                + parts_[-2]
                + "/"
                + str(id)
                + "_label_flip_"
                + parts_[-2]
                + "_"
                + str(parts_dot[0])
                + ".npy",
                label_image_flip,
            )
            np.save(
                output_dir
                + parts_[-2]
                + "/"
                + str(id)
                + "_input_augment2_"
                + parts_[-2]
                + "_"
                + str(parts_dot[0])
                + ".npy",
                input_data_augment_flip,
            )
