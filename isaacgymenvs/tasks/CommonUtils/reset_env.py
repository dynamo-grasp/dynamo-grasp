import random
import numpy as np
import torch
from homogeneous_transformation_and_conversion.rotation_conversions import *
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from pathlib import Path


class EnvReset:
    def __init__(self, gym, sim, physics_engine):
        self.gym = gym
        self.sim = sim
        self.physics_engine = physics_engine

    def random_number_with_probabilities(self, probabilities):
        random_number = random.random()
        for i, probability in enumerate(probabilities):
            if random_number < probability:
                return i
        return len(probabilities) - 1

    def reset_init_arm_pose(self, env_ids):
        for env_count in env_ids:
            env_count = env_count.item()
            # Calculating the random poses and random number of objects for the particular environment
            probabilities = self.object_bin_prob_spawn
            random_number = self.random_number_with_probabilities(probabilities)
            random_number += 1
            object_list_env = {}
            object_set = range(1, self.object_count_unique + 1)

            selected_object = np.random.choice(
                object_set, p=None, size=random_number, replace=False
            )

            list_objects_domain_randomizer = torch.tensor([])
            for object_count in selected_object:
                domain_randomizer = random_number = random.choice([1, 2, 3, 4, 5])
                # Spawning objects with random poses, but height is fixed
                offset_object = np.array(
                    [
                        np.random.uniform(0.67, 0.7, 1).reshape(
                            1,
                        )[0],
                        np.random.uniform(-0.22, -0.12, 1).reshape(
                            1,
                        )[0],
                        self.object_height_spawn,
                        np.random.uniform(0.0, 6.28, 1).reshape(
                            1,
                        )[0],
                        np.random.uniform(0.0, 6.28, 1).reshape(
                            1,
                        )[0],
                        np.random.uniform(0.0, 6.28, 1).reshape(
                            1,
                        )[0],
                    ]
                )
                quat = euler_angles_to_quaternion(
                    torch.tensor(offset_object[3:6]), "XYZ", degrees=False
                )
                offset_object = np.concatenate([offset_object[:3], quat.cpu().numpy()])
                item_config = (object_count - 1) * 5 + domain_randomizer
                object_list_env[item_config] = torch.tensor(offset_object)
                list_objects_domain_randomizer = torch.cat(
                    (list_objects_domain_randomizer, torch.tensor([item_config]))
                )
            self.selected_object_env[env_count] = list_objects_domain_randomizer
            self.object_pose_store[env_count] = object_list_env

        self.logger.info(
            f"Object configuration in each bin: {self.selected_object_env}"
        )
        pos = tensor_clamp(
            self.ur16e_default_dof_pos.unsqueeze(0),
            self.ur16e_dof_lower_limits.unsqueeze(0),
            self.ur16e_dof_upper_limits,
        )
        pos = pos.repeat(len(env_ids), 1)

        # Reinitializing the variables for resetting the environment
        for env_id in env_ids:
            self.action_contrib[env_id] = 2
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.free_envs_list[env_id] = torch.tensor(1)
            self.object_pose_check_list[env_id] = self.RETRY_OBJECT_STABILITY
            self.ee_vel[env_id] = self.DEFAULT_EE_VEL

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        return pos

    def reset_init_object_state(self, object, env_ids, offset):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        for i in range(len(self.object_models)):
            if object == self.object_models[i]:
                this_object_state_all = self._init_object_model_state[i]

        # Indexes corresponding to envs we're still actively sampling for
        active_idx = torch.arange(num_resets, device=self.device)
        sampled_object_state[:, 3:7] = offset[:, 3:7]
        sampled_object_state[:, :3] = offset[:, :3]

        # Lastly, set these sampled values as the new init state
        this_object_state_all[env_ids, :] = sampled_object_state

    def reset_object_pose(self, env_ids):
        """
        Resetting object poses and quering object pose from the saved object pose
        """
        for counter in range(len(self.object_models)):
            self.object_poses = torch.zeros(0, 7).to(self.device)
            for env_count in env_ids:
                # resetting force list
                # randomly spawning objects
                retKey = self.object_pose_store[env_count.item()].get(counter + 1)
                if retKey is None:
                    object_pose_env = torch.tensor(
                        [[counter / 4, 1, 0.5, 0.0, 0.0, 0.0, 1.0]]
                    ).to(self.device)
                else:
                    object_pose_env = (
                        self.object_pose_store[env_count.item()][counter + 1]
                        .clone()
                        .detach()
                        .to(self.device)
                    )
                    object_pose_env = object_pose_env.unsqueeze(0)
                self.object_poses = torch.cat([self.object_poses, object_pose_env])

            self.reset_init_object_state(
                env_ids=env_ids,
                object=self.object_models[counter],
                offset=self.object_poses,
            )
        # Resetting variables for resetting the object poses
        for env_count in env_ids:
            self.force_list_save[env_count.item()] = None
            self.target_object_disp_save[env_count.item()] = None
            self.object_disp_save[env_count.item()] = {}
            self.all_objects_last_pose[env_count.item()] = {}

        # setting the objects with randomly generated poses
        for counter in range(len(self.object_models)):
            self._object_model_state[counter][env_ids] = self._init_object_model_state[
                counter
            ][env_ids]
        # Setting the object poses in the simulation
        multi_env_ids_cubes_int32 = self._global_indices[
            env_ids, -len(self.object_models) :
        ].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32),
            len(multi_env_ids_cubes_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # reinitializing the variables
        for env_id in env_ids:
            self.action_contrib[env_id] = 2
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.env_reset_id_env[env_id] = 1
            self.ee_vel[env_id] = 0.15
            self.force_contact_flag[env_id.item()] = torch.tensor(0).type(torch.bool)

        self.object_movement_enabled = 0
        self.cmd_limit = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device
        ).unsqueeze(0)

    def reset_random_pick_pose(self, env_ids):
        """
        Resetting the pre grasp pose by randomly selecting the pre grasp pose from the
        list of pre grasp poses (for now there are only 2 pre grasp poses)
        """
        pos = torch.zeros(0, self.num_ur16e_dofs).to(self.device)
        for _ in env_ids:
            path = str(Path(__file__).parent.absolute())
            joint_poses_list = torch.load(f"{path}/../../../misc/joint_poses.pt")
            temp_pos = joint_poses_list[
                torch.randint(0, len(joint_poses_list), (1,))[0]
            ].to(self.device)
            temp_pos = torch.reshape(temp_pos, (1, len(temp_pos)))
            temp_pos = torch.cat((temp_pos, torch.tensor([[0]]).to(self.device)), dim=1)
            pos = torch.cat([pos, temp_pos])
        return pos

    def reset_idx_init(self, env_ids):
        """
        Resetting the arm to the initial arm pose and reset the object poses
        """
        pos = self.reset_init_arm_pose(env_ids)
        self.deploy_actions(env_ids, pos)
        self.reset_object_pose(env_ids)

    def reset_env_with_log(self, env_count, message, env_complete_reset):
        self.logger.info(message)
        env_complete_reset = torch.cat(
            (env_complete_reset, torch.tensor([env_count])), axis=0
        )
        self.free_envs_list[env_count] = torch.tensor(0)
        return env_complete_reset

    def reset_env_conditions(
        self, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset
    ):
        """
        Resetting the arm to a pre-grasp pose, adjusting object pose/configuration, or resetting a specific environment
        """
        if len(env_complete_reset) != 0 and len(env_list_reset_arm_pose) != 0:
            env_complete_reset = torch.unique(env_complete_reset)

            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, env_complete_reset), axis=0
            )

            env_list_reset_arm_pose = torch.unique(env_list_reset_arm_pose)
            env_list_reset_arm_pose = torch.tensor(
                [x for x in env_list_reset_arm_pose if x not in env_complete_reset]
            )

            env_ids = torch.cat((env_list_reset_arm_pose, env_complete_reset), axis=0)
            env_ids = env_ids.to(self.device).type(torch.long)
            pos1 = self.reset_random_pick_pose(
                env_list_reset_arm_pose.to(self.device).type(torch.long)
            )
            pos2 = self.reset_init_arm_pose(
                env_complete_reset.to(self.device).type(torch.long)
            )

            pos = torch.cat([pos1, pos2])
            self.deploy_actions(env_ids, pos)

        elif len(env_list_reset_arm_pose) != 0:
            env_list_reset_arm_pose = torch.unique(env_list_reset_arm_pose)
            pos = self.reset_random_pick_pose(
                env_list_reset_arm_pose.to(self.device).type(torch.long)
            )

            env_ids = env_list_reset_arm_pose.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)

        elif len(env_complete_reset) != 0:
            env_complete_reset = torch.unique(env_complete_reset)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, env_complete_reset), axis=0
            )
            pos = self.reset_init_arm_pose(
                env_complete_reset.to(self.device).type(torch.long)
            )
            env_ids = env_complete_reset.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)

        if len(env_list_reset_objects) != 0:
            env_list_reset_objects = torch.unique(env_list_reset_objects)
            self.reset_object_pose(
                env_list_reset_objects.to(self.device).type(torch.long)
            )

    def check_reset_conditions(self, env_count, env_complete_reset):
        segmask = self.get_segmask(env_count, camera_id=0)

        # Cropping bin area from the segmask
        segmask_bin_crop = segmask[
            self.check_object_coord[0] : self.check_object_coord[1],
            self.check_object_coord[2] : self.check_object_coord[3],
        ]
        # Calculating the number of objects spawned in the bin within the boundaries of the bin
        objects_spawned = len(torch.unique(segmask_bin_crop)) - 1
        total_objects = len(self.selected_object_env[env_count])
        object_coords_match = torch.count_nonzero(segmask) == torch.count_nonzero(
            segmask_bin_crop
        )
        total_position_error, obj_drop_status = self.calculate_objects_position_error(
            env_count
        )
        object_mask_area = self.min_area_object(segmask, segmask_bin_crop)
        # Check conditions for resetting the environment.
        conditions_messages = [
            (
                object_mask_area < 1000,
                f"Object in environment {env_count} not visible in the camera (due to occlusion) \
                    with area {object_mask_area}",
            ),
            (
                (not object_coords_match) or (total_objects != objects_spawned),
                f"Object in environment {env_count} extends beyond the bin's boundaries",
            ),
            (
                torch.abs(total_position_error) > self.POSE_ERROR_THRESHOLD,
                f"Object in environment {env_count}, not visible in the camera (due to occlusion) \
                    with area {object_mask_area}",
            ),
            (
                obj_drop_status,
                f"Object falled down in environment {env_count}, where total objects are {total_objects} \
                    and only {objects_spawned} were spawned inside the bin",
            ),
        ]
        for condition, message in conditions_messages:
            if condition:
                env_complete_reset = self.reset_env_with_log(
                    env_count, message, env_complete_reset
                )
                break
        return segmask, env_complete_reset
