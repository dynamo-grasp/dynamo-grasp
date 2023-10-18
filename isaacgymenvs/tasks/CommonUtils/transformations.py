from isaacgym import gymtorch
import torch
from homogeneous_transformation_and_conversion.rotation_conversions import *


class StaticDynamicTransformations:
    def __init__(self) -> None:
        pass

    def transformation_static_dynamic(self, env_count):
        """
        Transformation for static links
        """
        poses_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.curr_poses = gymtorch.wrap_tensor(poses_tensor).view(self.num_envs, -1, 13)
        # Transformation of base_link from world coordiante frame (wb)
        rotation_matrix_base_link = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        translation_base_link = torch.tensor([0, 0, 2.020]).to(self.device)
        self.T_base_link = transformation_matrix(
            rotation_matrix_base_link, translation_base_link
        )
        # Transformation for camera (wc --> wb*bc)
        rotation_matrix_camera_offset = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        T_base_link_to_camera = transformation_matrix(
            rotation_matrix_camera_offset, self.camera_base_link_translation
        )
        self.T_world_to_camera_link = torch.matmul(
            self.T_base_link, T_base_link_to_camera
        )

        """
        Transformation for dynamic links
        """
        # Transformation for object from camera (wo --> wc*co)
        rotation_matrix_camera_to_object = euler_angles_to_matrix(
            self.grasp_angle[env_count].to(self.device),
            "XYZ",
            degrees=False,
        )
        T_camera_to_object = transformation_matrix(
            rotation_matrix_camera_to_object, self.xyz_point[env_count]
        )
        # Transformation from base link to object
        self.T_world_to_object = torch.matmul(
            self.T_world_to_camera_link, T_camera_to_object
        )
        # Transformation for pre grasp pose (wp --> wo*op)
        rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(
            torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        translation_pre_grasp_pose = torch.tensor([-0.25, 0, 0]).to(self.device)
        T_pre_grasp_pose = transformation_matrix(
            rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose
        )
        # Transformation of object with base link to pre grasp pose
        self.T_world_to_pre_grasp_pose = torch.matmul(
            self.T_world_to_object, T_pre_grasp_pose
        )
        # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
        rotation_matrix_ee_pose = quaternion_to_matrix(
            self.curr_poses[env_count][self.multi_body_idx["ee_link"]][3:7]
        )
        translation_ee_pose = self.curr_poses[env_count][
            self.multi_body_idx["wrist_3_link"]
        ][:3]
        self.T_world_to_ee_pose = transformation_matrix(
            rotation_matrix_ee_pose, translation_ee_pose
        )
        self.T_ee_pose_to_pre_grasp_pose = torch.matmul(
            torch.inverse(self.T_world_to_ee_pose), self.T_world_to_pre_grasp_pose
        )
        # Orientation error
        self.action_orientation = matrix_to_euler_angles(
            self.T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ"
        )
