# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np

from autolab_core import (
    YamlConfig,
    Logger,
    ColorImage,
    RgbdImage,
)
from visualization import Visualizer2D as vis
from gqcnn.gqcnn.grasping import (
    RobustGraspingPolicy,
    RgbdImageState,
)

from pathlib import Path

cur_path = str(Path(__file__).parent.absolute())


class dexnet3:
    def __init__(self, cam_intr):
        self.camera_intr = cam_intr
        self.config_filename = cur_path + "/../cfg/examples/gqcnn_suction.yaml"
        self.model_dir = cur_path + "/../models"
        self.model_name = "GQCNN-3.0"
        self.logger = Logger.get_logger(
            cur_path + "/../examples/policy_for_training.py"
        )

    def load_dexnet_model(self):
        # Get configs
        model_path = os.path.join(self.model_dir, self.model_name)

        # Read config.
        self.config = YamlConfig(self.config_filename)
        self.policy_config = self.config["policy"]

        # Make relative paths absolute.
        if "gqcnn_model" in self.policy_config["metric"]:
            self.policy_config["metric"]["gqcnn_model"] = model_path
            if not os.path.isabs(self.policy_config["metric"]["gqcnn_model"]):
                self.policy_config["metric"]["gqcnn_model"] = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    self.policy_config["metric"]["gqcnn_model"],
                )

        self.policy = RobustGraspingPolicy(self.policy_config)

    def inference(self, depth_image, segmask, rgb_img):
        self.depth_im = depth_image
        self.segmask = segmask
        self.rgb_im_filename = rgb_img

        # Read images.
        color_im = ColorImage(
            np.zeros([self.depth_im.height, self.depth_im.width, 3]).astype(np.uint8),
            frame=self.camera_intr.frame,
        )

        # Optionally read a segmask.
        valid_px_mask = self.depth_im.invalid_pixel_mask().inverse()
        self.segmask = self.segmask.mask_binary(valid_px_mask)

        # Inpaint.
        inpaint_rescale_factor = self.config["inpaint_rescale_factor"]
        self.depth_im = self.depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

        if (
            "input_images" in self.policy_config["vis"]
            and self.policy_config["vis"]["input_images"]
        ):
            vis.figure(size=(10, 10))
            num_plot = 1
            if self.segmask is not None:
                num_plot = 2
            vis.subplot(1, num_plot, 1)
            vis.imshow(self.depth_im)
            if self.segmask is not None:
                vis.subplot(1, num_plot, 2)
                vis.imshow(self.segmask)
            vis.show()

        # Create state.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, self.depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=self.segmask)

        # Query policy.
        policy_start = time.time()
        # action = policy(state)

        action, grasps, q_values = self.policy(state)
        num_grasps = len(grasps)
        grasps_and_predictions = zip(grasps, q_values)
        grasps_and_predictions = sorted(
            grasps_and_predictions, key=lambda x: x[1], reverse=True
        )
        unsorted_grasps_and_predictions = list(zip(grasps, q_values))

        # Vis final grasp.
        # policy_config["vis"]["final_grasp"] = 0
        if self.policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(
                rgbd_im.depth,
                vmin=self.policy_config["vis"]["vmin"],
                vmax=self.policy_config["vis"]["vmax"],
            )
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title(
                "Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                    action.grasp.depth, action.q_value
                )
            )
            vis.show()

        return action, grasps_and_predictions, unsorted_grasps_and_predictions
