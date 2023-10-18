import random
import torch
import numpy as np
from autolab_core import BinaryImage, DepthImage
import os
from PIL import Image


class SampleGrasp:
    def __init__(self) -> None:
        pass

    def get_suction_and_object_param(
        self,
        env_count,
        env_list_reset_arm_pose,
        env_list_reset_objects,
        env_complete_reset,
    ):
        """
        Getting grasp angle, suction deformation score, xyz point, grasp point, dexnet score
        and force score from the stored values calcualted after sampling grasp points
        """
        self.env_reset_id_env[env_count] = torch.tensor(0)
        self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)

        if (env_count in self.grasp_angle_env) and (
            len(self.grasp_angle_env[env_count]) != 0
        ):
            self.suction_deformation_score[
                env_count
            ] = self.suction_deformation_score_env[env_count][0]
            self.suction_deformation_score_env[
                env_count
            ] = self.suction_deformation_score_env[env_count][1:]
            self.grasp_angle[env_count] = self.grasp_angle_env[env_count][0]
            self.grasp_angle_env[env_count] = self.grasp_angle_env[env_count][1:]
            self.xyz_point[env_count] = self.xyz_point_env[env_count][0]
            self.xyz_point_env[env_count] = self.xyz_point_env[env_count][1:]
            self.grasp_point[env_count] = self.grasp_point_env[env_count][0]
            self.grasp_point_env[env_count] = self.grasp_point_env[env_count][1:]
            self.dexnet_score[env_count] = self.dexnet_score_env[env_count][0]
            self.dexnet_score_env[env_count] = self.dexnet_score_env[env_count][1:]
            self.force_SI[env_count] = self.force_SI_env[env_count][0]
            self.force_SI_env[env_count] = self.force_SI_env[env_count][1:]
        else:
            env_complete_reset = torch.cat(
                (env_complete_reset, torch.tensor([env_count])), axis=0
            )
        try:
            if (env_count in self.grasp_angle_env) and (
                torch.count_nonzero(self.xyz_point[env_count]) < 1
            ):
                # error due to illegal 3d coordinate
                self.logger.info(f"illegal 3d coordinate in environment {env_count}")
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
                )
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0
                )

                self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

        except Exception as error:
            env_complete_reset = torch.cat(
                (env_complete_reset, torch.tensor([env_count])), axis=0
            )
            self.logger.warning(
                f"illegal environment configuration in environment {env_count}, and the error is {error}"
            )

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset

    def dexnet_sample_node(
        self,
        env_count,
        segmask,
        env_list_reset_arm_pose,
        env_list_reset_objects,
        env_complete_reset,
    ):
        """
        Running DexNet 3.0 and extracting sample points after investigating the pose error after spawning
        """
        random_object_select = random.sample(
            self.selected_object_env[env_count].tolist(), 1
        )
        self.object_target_id[env_count] = (
            torch.tensor(random_object_select).to(self.device).type(torch.int)
        )
        depth_image = self.get_depth_image(env_count, camera_id=0)
        segmask_dexnet = segmask.clone().detach()
        self.segmask_save[env_count] = (
            segmask[
                self.crop_coord[0] : self.crop_coord[1],
                self.crop_coord[2] : self.crop_coord[3],
            ]
            .clone()
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        segmask_numpy = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))
        segmask_numpy_temp = np.zeros_like(
            segmask_dexnet.cpu().numpy().astype(np.uint8)
        )
        segmask_numpy_temp[
            segmask_dexnet.cpu().numpy().astype(np.uint8)
            == self.object_target_id[env_count].cpu().numpy()
        ] = 1
        segmask_numpy[
            segmask_dexnet.cpu().numpy().astype(np.uint8)
            == self.object_target_id[env_count].cpu().numpy()
        ] = 255
        segmask_dexnet = BinaryImage(
            segmask_numpy[
                self.crop_coord[0] : self.crop_coord[1],
                self.crop_coord[2] : self.crop_coord[3],
            ],
            frame=self.camera_intrinsics_back_cam.frame,
        )
        depth_image_dexnet = depth_image.clone().detach()
        noise_image = torch.normal(0, 0.0005, size=depth_image_dexnet.size()).to(
            self.device
        )
        depth_image_dexnet = depth_image_dexnet + noise_image
        depth_image_save_temp = depth_image_dexnet.clone().detach().cpu().numpy()
        self.depth_image_save[env_count] = depth_image_save_temp[
            self.crop_coord[0] : self.crop_coord[1],
            self.crop_coord[2] : self.crop_coord[3],
        ]
        # saving depth image and segmentation mask only once before sampling grasp points
        self.config_env_count[env_count] += torch.tensor(1).type(torch.int)

        env_number = env_count
        new_dir_path = os.path.join(self.data_path, f"{env_number}/")

        env_config = self.config_env_count[env_count].type(torch.int).item()

        save_dir_depth_npy = os.path.join(
            new_dir_path, f"depth_image_{env_number}_{env_config}.npy"
        )
        save_dir_segmask_npy = os.path.join(
            new_dir_path, f"segmask_{env_number}_{env_config}.npy"
        )
        with open(save_dir_depth_npy, "wb") as f:
            np.save(f, self.depth_image_save[env_count])
        with open(save_dir_segmask_npy, "wb") as f:
            np.save(f, self.segmask_save[env_count])

        # cropping the image and modifying depth to match the DexNet 3.0 input configuration
        # Adjusting depth for DexNet: adding dexnet_thresh_offset to ensure depth is between 0.5 and 0.7 m
        dexnet_thresh_offset = -0.5
        depth_image_dexnet += dexnet_thresh_offset
        pod_back_panel_distance = torch.max(
            depth_image[
                self.crop_coord[0] : self.crop_coord[1],
                self.crop_coord[2] : self.crop_coord[3],
            ]
        ).item()
        # To make the depth of the image ranging from 0.5m to 0.7m for valid configuration for DexNet 3.0
        depth_numpy = depth_image_dexnet.cpu().numpy()
        depth_numpy_temp = depth_numpy * segmask_numpy_temp
        depth_numpy_temp[depth_numpy_temp == 0] = (
            pod_back_panel_distance + dexnet_thresh_offset
        )
        depth_img_dexnet = DepthImage(
            depth_numpy_temp[
                self.crop_coord[0] : self.crop_coord[1],
                self.crop_coord[2] : self.crop_coord[3],
            ],
            frame=self.camera_intrinsics_back_cam.frame,
        )
        max_num_grasps = 0

        # Storing all the sampled grasp point and its properties from DexNet 3.0 in their respective buffers
        try:
            (
                action,
                self.grasps_and_predictions,
                self.unsorted_grasps_and_predictions,
            ) = self.dexnet_object.inference(depth_img_dexnet, segmask_dexnet, None)
            max_num_grasps = len(self.grasps_and_predictions)
            self.logger.info(
                f"For environment {env_count} the number of grasp samples were {max_num_grasps}"
            )
            self.suction_deformation_score_temp = torch.Tensor()
            self.xyz_point_temp = torch.empty((0, 3))
            self.grasp_angle_temp = torch.empty((0, 3))
            self.grasp_point_temp = torch.empty((0, 2))
            self.force_SI_temp = torch.Tensor()
            self.dexnet_score_temp = torch.Tensor()
            top_grasps = max_num_grasps if max_num_grasps <= 10 else 7
            for i in range(max_num_grasps):
                grasp_point = torch.tensor(
                    [
                        self.grasps_and_predictions[i][0].center.x,
                        self.grasps_and_predictions[i][0].center.y,
                    ]
                )

                depth_image_suction = depth_image.clone().detach()
                offset = torch.tensor([self.crop_coord[2], self.crop_coord[0]])
                (
                    suction_deformation_score,
                    xyz_point,
                    grasp_angle,
                ) = self.suction_score_object.calculator(
                    depth_image_suction,
                    segmask,
                    self.grasps_and_predictions[i][0],
                    self.object_target_id[env_count],
                    offset,
                )
                grasp_angle = torch.tensor([0, 0, 0])
                self.suction_deformation_score_temp = torch.cat(
                    (
                        self.suction_deformation_score_temp,
                        torch.tensor([suction_deformation_score]),
                    )
                ).type(torch.float)
                self.xyz_point_temp = torch.cat(
                    [self.xyz_point_temp, xyz_point.unsqueeze(0)], dim=0
                )
                self.grasp_angle_temp = torch.cat(
                    [self.grasp_angle_temp, grasp_angle.unsqueeze(0)], dim=0
                )
                self.grasp_point_temp = torch.cat(
                    [self.grasp_point_temp, grasp_point.clone().detach().unsqueeze(0)],
                    dim=0,
                )
                self.object_coordiante_camera = xyz_point.clone().detach()
                if suction_deformation_score > 0:
                    force_SI = self.force_object.regression(suction_deformation_score)
                else:
                    force_SI = torch.tensor(0).to(self.device)

                self.force_SI_temp = torch.cat(
                    (self.force_SI_temp, torch.tensor([force_SI]))
                )
                self.dexnet_score_temp = torch.cat(
                    (
                        self.dexnet_score_temp,
                        torch.tensor([self.grasps_and_predictions[i][1]]),
                    )
                )

            if top_grasps > 0:
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
                )
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0
                )
            else:
                self.logger.error("No sample points")
                env_complete_reset = torch.cat(
                    (env_complete_reset, torch.tensor([env_count])), axis=0
                )
        except Exception as e:
            self.logger.error(f"dexnet error: {e}")
            env_complete_reset = torch.cat(
                (env_complete_reset, torch.tensor([env_count])), axis=0
            )

        self.suction_deformation_score_env[
            env_count
        ] = self.suction_deformation_score_temp
        self.grasp_angle_env[env_count] = self.grasp_angle_temp
        self.force_SI_env[env_count] = self.force_SI_temp
        self.xyz_point_env[env_count] = self.xyz_point_temp
        self.grasp_point_env[env_count] = self.grasp_point_temp
        self.dexnet_score_env[env_count] = self.dexnet_score_temp
        self.free_envs_list[env_count] = torch.tensor(0)

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset
