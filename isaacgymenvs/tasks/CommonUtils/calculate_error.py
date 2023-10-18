import torch
from homogeneous_transformation_and_conversion.rotation_conversions import *
import pandas as pd
from isaacgymenvs.utils.torch_jit_utils import *


class ErrorCalculator:
    def __init__(self) -> None:
        pass

    def reset_until_valid(
        self,
        env_count,
        env_list_reset_arm_pose,
        env_list_reset_objects,
        env_complete_reset,
    ):
        """
        Method to reset environments until a valid configuration is found.
        """
        if (
            self.frame_count[env_count] == self.COOLDOWN_FRAMES
        ) and self.object_pose_check_list[env_count]:
            # setting the pose of the object after cool down period
            self.object_pose_store[env_count] = self.store_objects_current_pose(
                env_count
            )
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0
            )
            self.object_pose_check_list[env_count] -= torch.tensor(1)

        # Spawning objects until they acquire stable pose and also doesn't falls down
        if (
            (self.frame_count[env_count] == self.COOLDOWN_FRAMES)
            and (not self.object_pose_check_list[env_count])
            and (self.free_envs_list[env_count] == torch.tensor(1))
        ):
            segmask, env_complete_reset = self.check_reset_conditions(
                env_count, env_complete_reset
            )

            # Check if the environment returned from reset and the frame for that enviornment is
            # equal to the COOLDOWN_FRAMES
            if env_count not in env_complete_reset:
                # Running DexNet 3.0 after investigating the pose error of the objects for sampling
                # grasp points
                (
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                ) = self.dexnet_sample_node(
                    env_count,
                    segmask,
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                )

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset

    def get_object_current_pose(self, env_count, object_id):
        """
        Retrieve the current pose of a specific object.
        """
        object_actor_index = self._object_model_id[int(object_id.item()) - 1]
        return (
            self._root_state[env_count, object_actor_index, :][:7]
            .type(torch.float)
            .clone()
            .detach()
        )

    def min_area_object(self, segmask_check, segmask_bin_crop):
        """
        Find the object with the minimum area in the segmask.
        """
        areas = [
            (segmask_check == int(object_id.item())).sum()
            for object_id in torch.unique(segmask_bin_crop)
        ]
        object_mask_area = min(areas, default=torch.tensor(1000))
        return object_mask_area

    def check_position_error(
        self,
        env_count,
        env_list_reset_arm_pose,
        env_list_reset_objects,
        env_complete_reset,
    ):
        """
        Check the position error of objects within the environment and determine if a reset is required.
        """
        _all_objects_current_pose = {}
        _all_object_position_error = torch.tensor(0.0).to(self.device)
        _all_object_rotation_error = torch.tensor(0.0).to(self.device)
        # collecting pose of all objects
        for object_id in self.selected_object_env[env_count]:
            _all_objects_current_pose[int(object_id.item())] = (
                self._root_state[
                    env_count, self._object_model_id[int(object_id.item()) - 1], :
                ][:3]
                .type(torch.float)
                .detach()
                .clone()
            )
            _all_object_position_error += torch.sum(
                self.object_pose_store[env_count][int(object_id.item())][:3]
                - self._root_state[
                    env_count, self._object_model_id[int(object_id.item()) - 1], :
                ][:3]
            )
            q1 = self.object_pose_store[env_count][int(object_id.item())][3:7]
            e1 = quaternion_to_euler_angles(q1, "XYZ", False)
            q2 = (
                self._root_state[
                    env_count, self._object_model_id[int(object_id.item()) - 1], :
                ][3:7]
                .type(torch.float)
                .detach()
                .clone()
            )
            e2 = quaternion_to_euler_angles(q2, "XYZ", False)
            _all_object_rotation_error += torch.sum(e1 - e2)

            if _all_objects_current_pose[int(object_id.item())][2] < torch.tensor(0.5):
                env_complete_reset = torch.cat(
                    (env_complete_reset, torch.tensor([env_count])), axis=0
                )
        _all_object_position_error = torch.abs(_all_object_position_error)
        _all_object_rotation_error = torch.abs(_all_object_rotation_error)
        if (_all_object_position_error > torch.tensor(0.0055)) and (
            self.action_contrib[env_count] == 2
        ):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
            )
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0
            )

        return (
            env_list_reset_arm_pose,
            env_list_reset_objects,
            env_complete_reset,
            _all_objects_current_pose,
        )

    def check_other_object_error(
        self,
        env_count,
        env_list_reset_arm_pose,
        env_list_reset_objects,
        _all_objects_current_pose,
    ):
        """
        Detecting interactions with incidental objects prior to reaching the target object.
        """
        _all_object_pose_error = torch.tensor(0.0).to(self.device)
        try:
            # estimating movement of other objects
            for object_id in self.selected_object_env[env_count]:
                if object_id != self.object_target_id[env_count]:
                    _all_object_pose_error += torch.abs(
                        torch.norm(
                            _all_objects_current_pose[int(object_id.item())][:3]
                            - self.all_objects_last_pose[env_count][
                                int(object_id.item())
                            ][:3]
                        )
                    )
        except Exception as error:
            _all_object_pose_error = torch.tensor(0.0)

        # reset if object has moved even before having contact with the target object
        if _all_object_pose_error > torch.tensor(0.0075):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
            )
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0
            )
            self.logger.info(
                f"{env_count} environment reset because of object collision before contact with target object",
            )
            self.save_config_grasp_json(env_count, False, torch.tensor(0), True)
        # Storing current pose of all objects
        for object_id in self.selected_object_env[env_count]:
            self.all_objects_last_pose[env_count][
                int(object_id.item())
            ] = _all_objects_current_pose[int(object_id.item())]

        return env_list_reset_arm_pose, env_list_reset_objects

    def store_objects_current_pose(self, env_count):
        bin_objects_current_pose = {}
        for object_id in self.selected_object_env[env_count]:
            bin_objects_current_pose[
                int(object_id.item())
            ] = self.get_object_current_pose(env_count, object_id)
        return bin_objects_current_pose

    def calculate_objects_position_error(self, env_count):
        """
        Check objects pose error with respect to the stored pose.
        """
        total_position_error = torch.tensor(0.0).to(self.device)
        for object_id in self.selected_object_env[env_count]:
            curr_pose = self.get_object_current_pose(env_count, object_id)[:3]
            stored_pose = self.object_pose_store[env_count][int(object_id.item())][:3]
            total_position_error += torch.sum(stored_pose - curr_pose)
            if curr_pose[2] < torch.tensor(0.5):
                return total_position_error, True
        return total_position_error, False

    def detect_contact_non_target_object(
        self, env_count, _all_objects_current_pose, contact_exist
    ):
        """
        Identify and manage unintended interactions with non-target objects.

        This function monitors the movement of all non-target objects within
        a given environment. If movement is detected without contact with the
        target object, it resets the environment using the store pose.
        """
        _all_object_pose_error = torch.tensor(0.0).to(self.device)
        try:
            # estimating movement of other objects
            for object_id in self.selected_object_env[env_count]:
                if object_id != self.object_target_id[env_count]:
                    _all_object_pose_error += torch.abs(
                        torch.norm(
                            _all_objects_current_pose[int(object_id.item())][:3]
                            - self.all_objects_last_pose[env_count][
                                int(object_id.item())
                            ][:3]
                        )
                    )
        except Exception as error:
            _all_object_pose_error = torch.tensor(0.0)

        # reset if object has moved even before having contact with the target object
        if (
            _all_object_pose_error > torch.tensor(0.0075)
        ) and contact_exist == torch.tensor(0):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
            )
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0
            )
            self.logger.info(
                f"Object in environment {env_count} moved without contact to target object by {_all_object_pose_error:2f} meters"
            )

            self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

    def calculate_angle_error(
        self, env_count, env_list_reset_arm_pose, env_list_reset_objects
    ):
        """
        detect the arm angle insertion error and reset the environment if the arm is not
        in the correct orientation with respect to pre grasp psoe
        """
        if self.action_contrib[env_count] == 0:
            angle_error = quaternion_to_euler_angles(
                self._eef_state[env_count][3:7], "XYZ", degrees=False
            ) - self.grasp_angle[env_count].to(self.device)
            if torch.max(torch.abs(angle_error)) > torch.deg2rad(torch.tensor(10.0)):
                # encountered the arm insertion constraint
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
                )
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0
                )
                self.logger.info(
                    f"{env_count} environment reset because of arm insertion angular constraint"
                )

                self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

            self.force_contact_flag[env_count] = torch.tensor(1).type(torch.bool)
        return env_list_reset_arm_pose, env_list_reset_objects

    def detect_target_object_movement(self, env_count, _all_objects_current_pose):
        """
        Monitor and react to the target object's movement during grasping.
        """
        current_object_pose = (
            self._root_state[
                env_count,
                self._object_model_id[self.object_target_id[env_count] - 1],
                :,
            ][:3]
            .type(torch.float)
            .detach()
            .clone()
        )
        # Calculate error by calculating the norm of the difference between the current and last object pose
        try:
            object_pose_error = torch.abs(
                torch.norm(current_object_pose - self.last_object_pose[env_count])
            )
        except:
            object_pose_error = torch.tensor(0)
        if object_pose_error >= 0.0003:
            self.object_movement_enabled = 1

        segmask_gripper = self.get_segmask(env_count, camera_id=1)

        depth_numpy_gripper = self.get_depth_image(env_count, camera_id=1)

        # Calculate the contact existence between the suction gripper and the target object surface
        try:
            contact_exist = self.suction_score_object_gripper.calculate_contact(
                depth_numpy_gripper,
                segmask_gripper,
                self.object_target_id[env_count],
            )
        except Exception as e:
            self.logger.warning(e)
            contact_exist = torch.tensor(0)
        # center pixel of the griper camera
        mask_point_cam = segmask_gripper[
            int(self.height_gripper / 2), int(self.width_gripper / 2)
        ]
        if mask_point_cam == self.object_target_id[env_count]:
            depth_point_cam = depth_numpy_gripper[
                int(self.height_gripper / 2), int(self.width_gripper / 2)
            ]
        else:
            depth_point_cam = torch.tensor(10.0)

        # If the object is moving then increase the ee_vel, else use the DEFAULT_EE_VEL value
        if (
            (depth_point_cam < torch.tensor(0.03))
            and (self.action_contrib[env_count] == torch.tensor(0))
            and (object_pose_error <= torch.tensor(0.001))
            and (contact_exist == torch.tensor(1))
        ):
            self.ee_vel[env_count] += torch.tensor(0.025)
            self.ee_vel[env_count] = torch.min(
                torch.tensor(1.0), self.ee_vel[env_count]
            )
            self.cmd_limit = to_torch(
                [0.25, 0.25, 0.25, 0.75, 0.75, 0.75], device=self.device
            ).unsqueeze(0)
        else:
            self.ee_vel[env_count] = self.DEFAULT_EE_VEL
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device
            ).unsqueeze(0)

        self.last_object_pose[env_count] = current_object_pose

        for object_id in self.selected_object_env[env_count]:
            self.all_objects_last_pose[env_count][
                int(object_id.item())
            ] = _all_objects_current_pose[int(object_id.item())]

        return contact_exist

    def detect_oscillation(self, force_list):
        """
        Detecting oscillations while grasping due to motion planning of the arm.
        """
        try:
            force = pd.DataFrame(force_list)
            force = force.astype(float)
            force_z_average = force.rolling(window=10).mean()
            force_numpy = force_z_average.to_numpy()
            dx = np.gradient(np.squeeze(force_numpy))
            dx = dx[~np.isnan(dx)]
            if np.min(dx) < -0.8:
                return True
            else:
                return False
        except:
            return False
