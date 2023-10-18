from isaacgymenvs.tasks.base.vec_task import VecTask
import os
import torch
import yaml
import assets.urdf_models.models_data as md
from isaacgym.torch_utils import *
from isaacgym import gymapi


class InitVariablesConfigs(VecTask):
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
        # Configuration and path assignments
        self.cfg = cfg
        self.data_path = data_path
        # Episode and action settings from configuration
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Bin and control type assignments
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {
            "osc",
            "joint_tor",
        }, "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # Setting observation and action dimensions based on control type
        self.cfg["env"]["numObservations"] = 9 if self.control_type == "osc" else 28
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 7

        # Initializing runtime variables, models, and tensor placeholders
        self.states = {}
        self.handles = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.models_lib = md.model_lib()

        # Tensor placeholders
        # State of root body (n_envs, 13)
        self._root_state = None
        self._dof_state = None  # State of all joints (n_envs, n_dof)
        self._q = None  # Joint positions (n_envs, n_dof)
        # Joint velocities (n_envs, n_dof)
        self._qd = None
        # State of all rigid bodies (n_envs, n_bodies, 13)
        self._rigid_body_state = None
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._ur16e_effort_limits = None  # Actuator effort limits for ur16e
        # Unique indices corresponding to all envs in flattened array
        self._global_indices = None

        # camera handles
        self.camera_handles = [[]]
        self.camera_info = None

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.ur16e_hand = "ee_link"
        self.ur16e_wrist_3_link = "wrist_3_link"
        self.ur16e_base = "base_link"
        self.suction_gripper = "epick_end_effector"

        self.force_threshold = 0.1
        self.object_coordiante_camera = torch.tensor([0, 0, 0])

        # Loading world parameters from a YAML file
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../configs")
            + "/pod_collision_primitives"
            + ".yml"
        ) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        # Invoking the VecTask's constructor
        VecTask.__init__(
            self,
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

    def init_reset_env_once(self):
        # Reset all environments
        self.reset_idx_init(torch.arange(self.num_envs, device=self.device))
        # Refresh tensors
        self.refresh_env_tensors()
        self.current_directory = os.getcwd()

    def initialize_var(self):
        # Constants
        self.DEFAULT_EE_VEL = torch.tensor(0.15)
        # default velocity of the gripper
        self.RETRY_OBJECT_STABILITY = torch.tensor(3)
        # number of retries for object stability
        self.COOLDOWN_FRAMES = 150
        # cooldown frames for objects to settle
        self.OBJECT_FALL_HEIGHT = torch.tensor(0.5)
        # all other object except from object used for the configuration is dropped from this height
        self.POSE_ERROR_THRESHOLD = torch.tensor(0.0055)
        # pose error threshold for object to be considered stable

        # Data path to store the data
        self.data_path = self.data_path or os.path.expanduser("~/temp/grasp_data/")
        for env_number in range(self.num_envs):
            new_dir_path = os.path.join(self.data_path, f"{env_number}/")
            os.makedirs(new_dir_path, exist_ok=True)

        # Establishing directories for storing data results
        self.force_list = np.array([])
        self.dexnet_coordinates = np.array([])
        self.grasp_angle = torch.tensor([[0, 0, 0]])
        self.grasps_and_predictions = None
        self.unsorted_grasps_and_predictions = None

        # OSC Gains
        self.kp = to_torch(
            [150.0, 150.0, 150.0, 100.0, 100.0, 100.0], device=self.device
        )
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.0] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Parameters of simulator
        self.action_contrib = torch.ones(self.num_envs) * 2
        self.frame_count_contact_object = torch.zeros(self.num_envs)
        self.force_encounter = torch.zeros(self.num_envs)
        self.frame_count = torch.zeros(self.num_envs)
        self.free_envs_list = torch.ones(self.num_envs)
        self.object_pose_check_list = torch.ones(self.num_envs)
        self.object_target_id = torch.zeros(self.num_envs).type(torch.int)
        self.ee_vel = torch.ones(self.num_envs) * self.DEFAULT_EE_VEL
        self.logger.info(f"No. of environments: {self.num_envs}")

        # Parameter storage and trackers for each environments
        self.suction_deformation_score_temp = torch.Tensor()
        self.xyz_point_temp = torch.Tensor([])
        self.grasp_angle_temp = torch.Tensor([])
        self.force_SI_temp = torch.Tensor()
        self.grasp_point_temp = torch.Tensor([])
        self.dexnet_score_temp = torch.Tensor([])

        self.suction_deformation_score_env = {}
        self.xyz_point_env = {}
        self.grasp_angle_env = {}
        self.force_SI_env = {}
        self.grasp_point_env = {}
        self.dexnet_score_env = {}
        self.force_contact_flag = torch.zeros(self.num_envs)

        self.suction_deformation_score = {}
        self.xyz_point = {}
        self.grasp_angle = {}
        self.force_SI = {}
        self.grasp_point = {}
        self.dexnet_score = {}
        self.last_object_pose = {}
        self.all_objects_last_pose = {}
        self.object_pose_store = {}
        self.target_object_disp_save = {}
        self.object_disp_save = {}

        self.selected_object_env = {}

        self.track_save = torch.zeros(self.num_envs)
        self.config_env_count = torch.zeros(self.num_envs)
        self.force_list_save = {}
        self.depth_image_save = {}
        self.segmask_save = {}
        self.grasp_point_save = {}
        self.grasp_angle_save = {}
        self.dexnet_score_save = {}
        self.suction_deformation_score_save = {}
        self.force_require_SI = {}

        self.env_reset_id_env = torch.ones(self.num_envs)

        # Defining bin specific configurations and control limits
        self.object_bin_prob_spawn = [0.05, 0.55, 1]
        self.object_height_spawn = 1.3

        self.ur16e_default_dof_pos = to_torch(
            [-1.57, 0, 0, 0, 0, 0, 0], device=self.device
        )
        self.check_object_coord = [335, 531, 646, 846]
        self.crop_coord = [180, 660, 410, 1050]

        # Set control limits
        self.cmd_limit = (
            to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
            if self.control_type == "osc"
            else self._ur16e_effort_limits[:6].unsqueeze(0)
        )

    def create_sim(self):
        """
        Defining simulation parameters and creating simulation environments
        """
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.create_ground_plane()
        self.create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    # TODO: Explain what it does
    def refresh_env_tensors(self):
        """
        Refreshing the tensors for the simulation environments
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        # Refresh states
        self.update_states()

    def update_states(self):
        """
        Updating states with current environment and robotic parameters
        """
        self.states.update(
            {
                # ur16e
                "base_link": self._base_link[:, :7],
                "wrist_3_link": self._wrist_3_link[:, :7],
                "wrist_3_link_vel": self._wrist_3_link[:, 7:],
                "q": self._q[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "eef_vel": self._eef_state[:, 7:],
            }
        )
