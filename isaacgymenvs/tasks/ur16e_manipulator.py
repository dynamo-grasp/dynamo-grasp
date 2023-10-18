import torch
import logging
from isaacgymenvs.utils.torch_jit_utils import *


from isaacgymenvs.tasks.CommonUtils.setup_env import (
    EnvSetup,
)
from isaacgymenvs.tasks.CommonUtils.reset_env import (
    EnvReset,
)  # Importing EnvReset, which provides functions encapsulating conditions for resetting the environment
from isaacgymenvs.tasks.CommonUtils.init_variables_configs import (
    InitVariablesConfigs,
)  # Importing InitVariablesConfigs, initializes and configures various variables and environment settings
from isaacgymenvs.tasks.CommonUtils.sample_grasp import (
    SampleGrasp,
)  # Importing SampleGrasp, which provides functions to sample grasp points using GQCNN
from isaacgymenvs.tasks.CommonUtils.robot_control import (
    RobotControl,
)  # Importing RobotControl, which provides functions to control the ur16e robotic arm
from isaacgymenvs.tasks.CommonUtils.calculate_error import (
    ErrorCalculator,
)  # ErrorCalculator is used to check different types of errors in the environment
from isaacgymenvs.tasks.CommonUtils.get_sensor_values import (
    GetSensorValues,
)  # GetSensorValues helps in getting sensor values from the environment
from isaacgymenvs.tasks.CommonUtils.saving_data import (
    SavingData,
)  # SavingData is used to save data from the environment in json files
from isaacgymenvs.tasks.CommonUtils.transformations import (
    StaticDynamicTransformations,
)  # StaticDynamicTransformations is used to calculate transformations for robot control
from isaacgymenvs.tasks.CommonUtils.evaluate_success import (
    EvlauateSuccess,
)  # EvlauateSuccess is used to evaluate the success of the grasp or collision of the robot with the environment


class UR16eManipulation(
    InitVariablesConfigs,
    EnvReset,
    EnvSetup,
    SampleGrasp,
    RobotControl,
    ErrorCalculator,
    GetSensorValues,
    SavingData,
    StaticDynamicTransformations,
    EvlauateSuccess,
):
    """
    The UR16eManipulation class combined functionalities provided by various utility classes
        to accomplish data collection using the UR16e robotic arm within a Isaac Gym simualtor.

    This class is responsible for initializing configurations, setting up and resetting the environment,
        sampling grasps, controlling the robot, calculating errors, retrieving sensor values,
        saving data, transforming static-dynamic elements, and evaluating task success.

    Methods:
        __init__: Initializes configurations, environments, and various utility classes.
        compute_observations: Computes observations, collecting specific states and concatenating them into an observation buffer.
        pre_physics_step: Performs pre-physics step operations, including checking and managing reset conditions,
            sampling grasp points, calculating transformations, and setting control inputs, among other functionalities.
        post_physics_step: Performs operations after the physics step, managing timeouts, saving configurations on reset,
            deploying actions, and resetting object poses, among other actions.

    Usage:
        - Use this class for collecting data by running the pipeline sequentially.
        - Utilize various inherited utility classes to manage environmental setup, control, error calculation, data retrieval,
            and other robot manipulation aspects.
    """

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
        data_path=None,
    ):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        # Create a file handler
        handler = logging.FileHandler('/tmp/dynamo_grasp_log.txt')
        self.logger = logging.getLogger("DYNAMO-GRASP")
        self.logger.addHandler(handler)

        InitVariablesConfigs.__init__(
            self,
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
            data_path,
        )
        EnvReset.__init__(self, self.gym, self.sim, self.physics_engine)
        EnvSetup.__init__(self, self.gym, self.sim, self.physics_engine)
        # Initialize all variables and constants, followed by a one-time environment reset
        self.initialize_var()
        self.init_reset_env_once()
        SampleGrasp.__init__(self)
        RobotControl.__init__(self)
        ErrorCalculator.__init__(self)
        GetSensorValues.__init__(self)
        StaticDynamicTransformations.__init__(self)
        EvlauateSuccess.__init__(self)

    def compute_observations(self):
        """
        Compile observations by concatenating end-effector states.
        """
        self.refresh_env_tensors()
        obs = ["eef_pos", "eef_quat", "q_gripper"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        return self.obs_buf

    def pre_physics_step(self, actions):
        """
        Perform preliminary physics calculations and apply actions prior to the simulation step.
        """
        self.refresh_real_time_sensors()
        self.actions = torch.zeros(0, 7)
        # Variables to track environments where reset conditions have been met
        env_list_reset_objects = torch.tensor([])
        env_list_reset_arm_pose = torch.tensor([])
        env_complete_reset = torch.tensor([])
        # Looping over all environments and for each environment, action is calculated and appended
        # to the action buffer according to the pipeline
        for env_count in range(self.num_envs):
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device
            ).unsqueeze(0)
            # Checking reset conditions until a valid configuration is found
            (
                env_list_reset_arm_pose,
                env_list_reset_objects,
                env_complete_reset,
            ) = self.reset_until_valid(
                env_count,
                env_list_reset_arm_pose,
                env_list_reset_objects,
                env_complete_reset,
            )
            # Reset action tensor to zero before calculating new action
            self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
            # Checking condition for grasp point sampling
            if (self.env_reset_id_env[env_count] == 1) and (
                self.frame_count[env_count] > self.COOLDOWN_FRAMES
            ):
                # Sample grasp points and calculate correponding suciton deformation score and required
                # force to grasp the object
                (
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                ) = self.get_suction_and_object_param(
                    env_count,
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                )
            # Checking condition for executing a suction grasp
            elif (
                self.env_reset_id_env[env_count] == 0
                and self.frame_count[env_count] > torch.tensor(self.COOLDOWN_FRAMES)
                and self.free_envs_list[env_count] == torch.tensor(0)
            ):
                # Append force values from the force sensor at 'wrist_3_link' and displacement values of the
                # target object to their respective buffers
                self.store_force_and_displacement(env_count)
                # Calculate transformations for robot control
                self.transformation_static_dynamic(env_count)
                (
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                    _all_objects_current_pose,
                ) = self.check_position_error(
                    env_count,
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                    env_complete_reset,
                )

                # Setting the control input for pregrasp pose by using the transformation
                # between the current end-effector pose and the pre-grasp pose
                if self.action_contrib[env_count] >= torch.tensor(1):
                    pose_factor, ori_factor = 1.0, 0.3
                    self.action_env = torch.tensor(
                        [
                            [
                                pose_factor * self.T_ee_pose_to_pre_grasp_pose[0][3],
                                pose_factor * self.T_ee_pose_to_pre_grasp_pose[1][3],
                                pose_factor * self.T_ee_pose_to_pre_grasp_pose[2][3],
                                ori_factor * self.action_orientation[0],
                                ori_factor * self.action_orientation[1],
                                ori_factor * self.action_orientation[2],
                                1,
                            ]
                        ],
                        dtype=torch.float,
                    )
                    # Reset environment if the configuration is unstable by checking position
                    # error of all objects except the target object
                    (
                        env_list_reset_arm_pose,
                        env_list_reset_objects,
                    ) = self.check_other_object_error(
                        env_count,
                        env_list_reset_arm_pose,
                        env_list_reset_objects,
                        _all_objects_current_pose,
                    )
                else:
                    # Execute the grasp action
                    self.calculate_grasp_action(env_count)
                    contact_exist = self.detect_target_object_movement(
                        env_count, _all_objects_current_pose
                    )
                    (
                        env_list_reset_arm_pose,
                        env_list_reset_objects,
                    ) = self.calculate_angle_error(
                        env_count, env_list_reset_arm_pose, env_list_reset_objects
                    )
                    self.detect_contact_non_target_object(
                        env_count, _all_objects_current_pose, contact_exist
                    )
                # Evaluating if the force threshold has been crossed and if the grasp is successful
                (
                    env_list_reset_arm_pose,
                    env_list_reset_objects,
                ) = self.evaluate_suction_grasp_success(
                    env_count, env_list_reset_arm_pose, env_list_reset_objects
                )

            if self.frame_count[env_count] <= torch.tensor(self.COOLDOWN_FRAMES):
                self.action_env = torch.tensor(
                    [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float
                )

            self.actions = torch.cat([self.actions, self.action_env])
            self.frame_count[env_count] += torch.tensor(1)

        # Here it resets the environment, arm pose and object pose, if the reset conditions are met
        self.reset_env_conditions(
            env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset
        )
        self.execute_control_actions()

    def post_physics_step(self):
        """
        Perform post-simulation step operations such as checking for timeout and resetting.
        """
        # Check if the environment has timed out
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # check if there is a timeout and reset the environment
        if len(env_ids) > 0:
            for env_id in env_ids:
                env_count = env_id.item()
                if (
                    self.force_list_save[env_count] != None
                    and len(self.force_list_save[env_count]) > 10
                ):
                    self.save_config_grasp_json(env_count, True, torch.tensor(0), False)
                else:
                    self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

            self.logger.info(f"timeout reset for environment {env_ids}")
            pos = self.reset_random_pick_pose(env_ids)
            self.deploy_actions(env_ids, pos)
            self.reset_object_pose(env_ids)

        self.compute_observations()

        # Compute possible resets
        self.reset_buf = torch.where(
            (self.progress_buf >= self.max_episode_length - 1),
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
