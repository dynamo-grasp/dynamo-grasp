import torch


class EvlauateSuccess:
    def __init__(self) -> None:
        pass

    def evaluate_suction_grasp_success(
        self, env_count, env_list_reset_arm_pose, env_list_reset_objects
    ):
        """
        Evaluate the success of a suction grasp attempt and also detect collision of the arm with the environment.
        """
        # If arm pushes the object such that it crosses the force threshold required to grasp the object
        if self.frame_count[env_count] > torch.tensor(
            self.COOLDOWN_FRAMES
        ) and self.frame_count_contact_object[env_count] == torch.tensor(0):
            if (torch.max(torch.abs(self.action_env[0][:3]))) <= 0.001 and (
                torch.max(torch.abs(self.action_env[0][3:6]))
            ) <= 0.001:
                self.action_contrib[env_count] -= 1

                # Gripper camera
                segmask_gripper = self.get_segmask(env_count, camera_id=1)
                depth_image = self.get_depth_image(env_count, camera_id=1)
                depth_numpy_gripper = depth_image.clone().detach()
                offset = torch.tensor([self.crop_coord[2], self.crop_coord[0]])
                (
                    self.suction_deformation_score[env_count],
                    temp_xyz_point,
                    temp_grasp_angle,
                ) = self.suction_score_object_gripper.calculator(
                    depth_numpy_gripper,
                    segmask_gripper,
                    None,
                    self.object_target_id[env_count],
                    offset,
                )

                if self.suction_deformation_score[env_count] > self.force_threshold:
                    self.force_SI[env_count] = self.force_object.regression(
                        self.suction_deformation_score[env_count]
                    )
                else:
                    self.force_SI[env_count] = torch.tensor(1000).to(self.device)
                if self.action_contrib[env_count] == 1:
                    self.xyz_point[env_count][0] += temp_xyz_point[0]
                    self.grasp_angle[env_count] = temp_grasp_angle

            if (
                self.force_pre_physics
                > torch.max(
                    torch.tensor([self.force_threshold, self.force_SI[env_count]])
                )
                and self.action_contrib[env_count] == 0
            ):
                self.force_encounter[env_count] = 1
                # Gripper camera
                segmask_gripper = self.get_segmask(env_count, camera_id=1)
                depth_image = self.get_depth_image(env_count, camera_id=1)
                depth_numpy_gripper = depth_image.clone().detach()
                offset = torch.tensor([self.crop_coord[2], self.crop_coord[0]])
                score_gripper, _, _ = self.suction_score_object_gripper.calculator(
                    depth_numpy_gripper,
                    segmask_gripper,
                    None,
                    self.object_target_id[env_count],
                    offset,
                )
                self.logger.info(
                    f"In environment {env_count},\n            force: {self.force_pre_physics},\n            suction deformation score: {score_gripper}"
                )
                # self.logger.info(f"environment {env_count}, suction deformation score: {score_gripper}")

                self.frame_count_contact_object[env_count] = 1

                self.save_config_grasp_json(env_count, True, score_gripper, False)

            # If the arm collided with the environment, reset the environment
            elif (
                self.force_pre_physics > torch.tensor(10)
                and self.action_contrib[env_count] == 1
            ):
                self.logger.info(
                    f"enviornment {env_count} reset, because of collision between arm and environment: ",
                    self.force_pre_physics,
                )
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
                )
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0
                )
                self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

        # This condition is to try next grasp point if the current grasp point is successfully grased
        elif self.frame_count_contact_object[env_count] == torch.tensor(
            1
        ) and self.frame_count[env_count] > torch.tensor(self.COOLDOWN_FRAMES):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0
            )
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0
            )

        return env_list_reset_arm_pose, env_list_reset_objects
