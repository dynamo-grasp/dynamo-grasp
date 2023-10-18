from homogeneous_transformation_and_conversion.rotation_conversions import *
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from autolab_core import CameraIntrinsics
from suction_cup_modelling.suction_score_calcualtor import calcualte_suction_score
from suction_cup_modelling.force_calculator import calcualte_force
from gqcnn.gqcnn_examples.policy_for_training import dexnet3


class EnvSetup:
    def __init__(self, gym, sim, physics_engine):
        self.gym = gym
        self.sim = sim
        self.physics_engine = physics_engine

    def create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_envs(self, num_envs, spacing, num_per_row):
        """
        Creates and initializes environments by loading the robot model, object models and table model.

        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "assets"
        ur16e_asset_file = "urdf/Aurmar_description/robots/ur16e.urdf"

        # load ur16e asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        ur16e_asset = self.gym.load_asset(
            self.sim, asset_root, ur16e_asset_file, asset_options
        )

        ur16e_dof_stiffness = to_torch(
            [0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device
        )
        ur16e_dof_damping = to_torch(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device
        )

        # Create table asset and load table model
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True

        # Load object models from the object list with domain randomization
        self.object_models = []
        objects_file = open(f"misc/object_list_domain_randomization.txt", "r")
        object_config = objects_file.readlines()

        objects = []
        for parameters in object_config:
            objects.append(parameters)

        self.object_count_unique = 0
        for object in objects:
            self.object_count_unique += 1
            for domain in range(5):
                self.object_models.append(str(object.strip()) + "_" + str(domain + 1))
        object_model_asset_file = []
        object_model_asset = []
        for counter, model in enumerate(self.object_models):
            object_model_asset_file.append(
                "urdf_models/models/" + model + "/model.urdf"
            )

            object_model_asset.append(
                self.gym.load_asset(
                    self.sim,
                    asset_root,
                    object_model_asset_file[counter],
                    asset_options,
                )
            )

        self.num_ur16e_bodies = self.gym.get_asset_rigid_body_count(ur16e_asset)
        self.num_ur16e_dofs = self.gym.get_asset_dof_count(ur16e_asset)

        self.logger.info(f"num ur16e bodies: {self.num_ur16e_bodies}")
        self.logger.info(f"num ur16e dofs: {self.num_ur16e_dofs}")

        # Set ur16e dof properties
        ur16e_dof_props = self.gym.get_asset_dof_properties(ur16e_asset)
        self.ur16e_dof_lower_limits = []
        self.ur16e_dof_upper_limits = []
        self._ur16e_effort_limits = []
        for i in range(self.num_ur16e_dofs):
            ur16e_dof_props["driveMode"][i] = (
                gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            )
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur16e_dof_props["stiffness"][i] = ur16e_dof_stiffness[i]
                ur16e_dof_props["damping"][i] = ur16e_dof_damping[i]
            else:
                ur16e_dof_props["stiffness"][i] = 7000.0
                ur16e_dof_props["damping"][i] = 50.0

            self.ur16e_dof_lower_limits.append(ur16e_dof_props["lower"][i])
            self.ur16e_dof_upper_limits.append(ur16e_dof_props["upper"][i])
            self._ur16e_effort_limits.append(ur16e_dof_props["effort"][i])

        self.ur16e_dof_lower_limits = to_torch(
            self.ur16e_dof_lower_limits, device=self.device
        )
        self.ur16e_dof_upper_limits = to_torch(
            self.ur16e_dof_upper_limits, device=self.device
        )
        self._ur16e_effort_limits = to_torch(
            self._ur16e_effort_limits, device=self.device
        )
        self.ur16e_dof_speed_scales = torch.ones_like(self.ur16e_dof_lower_limits)

        # Define start pose for ur16e
        ur16e_start_pose = gymapi.Transform()
        ur16e_start_pose.p = gymapi.Vec3(0, 0, 2.020)

        quat = euler_angles_to_quaternion(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        ur16e_start_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )

        object_model_start_pose = []
        for counter in range(len(self.object_models)):
            object_model_start_pose.append(gymapi.Transform())
            object_model_start_pose[counter].p = gymapi.Vec3(0.0, -0.1, -10.0)
            object_model_start_pose[counter].r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.table_count = 0
        if "cube" in self.world_params["world_model"]["coll_objs"]:
            cube = self.world_params["world_model"]["coll_objs"]["cube"]
            for obj in cube.keys():
                self.table_count += 1

        # Compute aggregate size for all assets
        num_ur16e_bodies = self.gym.get_asset_rigid_body_count(ur16e_asset)
        num_ur16e_shapes = self.gym.get_asset_rigid_shape_count(ur16e_asset)
        max_agg_bodies = num_ur16e_bodies + self.table_count + len(self.object_models)
        max_agg_shapes = num_ur16e_shapes + self.table_count + len(self.object_models)

        self.ur16es = []
        self.envs = []

        self.multi_body_idx = {
            "base_link": self.gym.find_asset_rigid_body_index(ur16e_asset, "base_link"),
            "wrist_3_link": self.gym.find_asset_rigid_body_index(
                ur16e_asset, "wrist_3_link"
            ),
            "ee_link": self.gym.find_asset_rigid_body_index(ur16e_asset, "ee_link"),
            "epick_end_effector": self.gym.find_asset_rigid_body_index(
                ur16e_asset, "epick_end_effector"
            ),
        }

        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(
            ur16e_asset, self.multi_body_idx["wrist_3_link"], sensor_pose
        )

        # Create environments and actors for each environment
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: ur16e should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create ur16e actor
            self.ur16e_actor = self.gym.create_actor(
                env_ptr, ur16e_asset, ur16e_start_pose, "ur16e", i, 0, 0
            )
            self.gym.set_actor_dof_properties(
                env_ptr, self.ur16e_actor, ur16e_dof_props
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, self.ur16e_actor)
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create pod using the cubes in the world model
            if "cube" in self.world_params["world_model"]["coll_objs"]:
                cube = self.world_params["world_model"]["coll_objs"]["cube"]
                for obj in cube.keys():
                    if int(obj[4:]) >= 100:
                        dims = cube[obj]["dims"]
                        pose = cube[obj]["pose"]
                        self.add_table(
                            dims,
                            pose,
                            ur16e_start_pose,
                            env_ptr,
                            i,
                            # color=[1.0, 0.96, 0.18],
                            color=[0.6, 0.6, 0.6],
                            mesh_visual_only=False,
                        )
                    else:
                        dims = cube[obj]["dims"]
                        pose = cube[obj]["pose"]
                        self.add_table(
                            dims,
                            pose,
                            ur16e_start_pose,
                            env_ptr,
                            i,
                            color=[0.6, 0.6, 0.6],
                            mesh_visual_only=False,
                        )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Spawn urdf primitive shapes objects
            self._object_model_id = []
            for counter in range(len(self.object_models)):
                self._object_model_id.append(
                    self.gym.create_actor(
                        env_ptr,
                        object_model_asset[counter],
                        object_model_start_pose[counter],
                        self.object_models[counter],
                        i,
                        0,
                        counter + 1,
                    )
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env and ur16e pointers
            self.envs.append(env_ptr)
            self.ur16es.append(self.ur16e_actor)

            # Adding friction to the suction gripper
            ur16e_handle = 0
            suction_gripper_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, ur16e_handle, self.suction_gripper
            )
            suction_gripper_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, suction_gripper_handle
            )
            # High friction for suction gripper to avoid slipping and fill the gap between
            # real robot and simulator
            suction_gripper_shape_props[0].friction = 1.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, suction_gripper_handle, suction_gripper_shape_props
            )

            # Camera environment setup (Back cam) and its properties
            self.camera_handles.append([])
            self.body_states = []
            self.camera_properties_back_cam = gymapi.CameraProperties()
            self.camera_properties_back_cam.enable_tensors = True
            self.camera_properties_back_cam.horizontal_fov = 70.0
            self.camera_properties_back_cam.width = 1280
            self.camera_properties_back_cam.height = 786
            camera_handle = self.gym.create_camera_sensor(
                env_ptr, self.camera_properties_back_cam
            )
            self.camera_base_link_translation = torch.tensor([-0.48, 0.05, 0.6]).to(
                self.device
            )
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(
                self.camera_base_link_translation[0],
                self.camera_base_link_translation[1],
                self.camera_base_link_translation[2],
            )
            camera_rotation_x = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(1, 0, 0), np.deg2rad(180)
            )
            local_transform.r = camera_rotation_x

            ur16e_base_link_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, 0, self.ur16e_base
            )
            self.gym.attach_camera_to_body(
                camera_handle,
                env_ptr,
                ur16e_base_link_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.camera_handles[i].append(camera_handle)

            # Setup embedded camera at gripper for real time feedback (Gripper cam)
            # and its properties
            ur16e_hand_link_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, 0, "ee_link"
            )
            self.camera_gripper_link_translation = []
            self.camera_properties_gripper = gymapi.CameraProperties()
            self.camera_properties_gripper.enable_tensors = True
            self.camera_properties_gripper.horizontal_fov = 150.0
            self.camera_properties_gripper.width = 1920
            self.camera_properties_gripper.height = 1080
            camera_handle_gripper = self.gym.create_camera_sensor(
                env_ptr, self.camera_properties_gripper
            )
            self.camera_gripper_link_translation.append(
                torch.tensor([0.0, 0, 0]).to(self.device)
            )
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(
                self.camera_gripper_link_translation[0][0],
                self.camera_gripper_link_translation[0][1],
                self.camera_gripper_link_translation[0][2],
            )
            self.gym.attach_camera_to_body(
                camera_handle_gripper,
                env_ptr,
                ur16e_hand_link_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.camera_handles[i].append(camera_handle_gripper)

            l_color = gymapi.Vec3(1, 1, 1)
            l_ambient = gymapi.Vec3(0.1, 0.1, 0.1)
            l_direction = gymapi.Vec3(-1, -1, 1)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            l_direction = gymapi.Vec3(-1, 1, 1)
            self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)

        # Setup data structures for rendering
        self.init_data()

        # List for storing the object poses
        self._init_object_model_state = []
        for counter in range(len(self.object_models)):
            self._init_object_model_state.append(
                torch.zeros(self.num_envs, 13, device=self.device)
            )

    def add_table(
        self,
        table_dims,
        table_pose,
        robot_pose,
        env_ptr,
        env_id,
        color=[1.0, 0.0, 0.0],
        mesh_visual_only=False,
    ):
        """
        Pod is setup by adding the cubes to form bins
        """
        table_dims = gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        obj_color = gymapi.Vec3(color[0], color[1], color[2])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(table_pose[0], table_pose[1], table_pose[2])
        pose.r = gymapi.Quat(table_pose[3], table_pose[4], table_pose[5], table_pose[6])
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )

        table_pose = robot_pose * pose
        table_handle = self.gym.create_actor(
            env_ptr, table_asset, table_pose, "table", env_id, 0
        )
        if not mesh_visual_only:
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color
            )
        else:
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, obj_color
            )
        table_shape_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, table_handle
        )
        table_shape_props[0].friction = 0.4
        self.gym.set_actor_rigid_shape_properties(
            env_ptr, table_handle, table_shape_props
        )

    def init_data(self):
        # Setup sim handles and object models for rendering
        env_ptr = self.envs[0]
        ur16e_handle = 0
        self.handles = {
            "base_link": self.gym.find_actor_rigid_body_handle(
                env_ptr, ur16e_handle, self.ur16e_base
            ),
            "wrist_3_link": self.gym.find_actor_rigid_body_handle(
                env_ptr, ur16e_handle, self.ur16e_wrist_3_link
            ),
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, ur16e_handle, self.ur16e_hand
            ),
        }

        for counter in range(len(self.object_models)):
            self.handles[
                self.object_models[counter] + "_body_handle"
            ] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self._object_model_id[counter],
                self.object_models[counter],
            )
            object_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, self._object_model_id[counter]
            )
            object_shape_props[0].friction = 0.2
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, self._object_model_id[counter], object_shape_props
            )

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._base_link = self._rigid_body_state[:, self.handles["base_link"], :]
        self._wrist_3_link = self._rigid_body_state[:, self.handles["wrist_3_link"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur16e")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, ur16e_handle)[
            "gripper_joint"
        ]
        self._j_eef = jacobian[:, hand_joint_index, :, :6]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur16e")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]

        self._object_model_state = []
        for counter in range(len(self.object_models)):
            self._object_model_state.append(
                self._root_state[:, self._object_model_id[counter], :]
            )

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)
        # Initialize control
        self._arm_control = self._effort_control[:, :]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(
            self.num_envs * (self.table_count + 1 + len(self.object_models)),
            dtype=torch.int32,
            device=self.device,
        ).view(self.num_envs, -1)

        # camera intrinsics for back camera and gripper camera
        cam_proj_back_cam = torch.tensor(
            self.gym.get_camera_proj_matrix(
                self.sim, self.envs[0], self.camera_handles[0][0]
            ),
            device=self.device,
        )
        self.width_back_cam = self.camera_properties_back_cam.width
        self.height_back_cam = self.camera_properties_back_cam.height
        self.fx_back_cam = self.width_back_cam / (2 / cam_proj_back_cam[0, 0])
        self.fy_back_cam = self.height_back_cam / (2 / cam_proj_back_cam[1, 1])
        self.cx_back_cam = self.width_back_cam / 2
        self.cy_back_cam = self.height_back_cam / 2
        self.camera_intrinsics_back_cam = CameraIntrinsics(
            frame="camera_back",
            fx=self.fx_back_cam,
            fy=self.fy_back_cam,
            cx=self.cx_back_cam,
            cy=self.cy_back_cam,
            skew=0.0,
            height=self.height_back_cam,
            width=self.width_back_cam,
        )
        self.suction_score_object = calcualte_suction_score(
            self.camera_intrinsics_back_cam
        )
        self.dexnet_object = dexnet3(self.camera_intrinsics_back_cam)
        self.dexnet_object.load_dexnet_model()

        self.logger.info(f"focal length in x axis: {self.fx_back_cam}")
        self.logger.info(f"focal length in y axis: {self.fy_back_cam}")
        cam_proj_gripper = torch.tensor(
            self.gym.get_camera_proj_matrix(
                self.sim, self.envs[0], self.camera_handles[0][1]
            ),
            device=self.device,
        )
        self.width_gripper = self.camera_properties_gripper.width
        self.height_gripper = self.camera_properties_gripper.height
        self.fx_gripper = self.width_gripper / (2 / cam_proj_gripper[0, 0])
        self.fy_gripper = self.height_gripper / (2 / cam_proj_gripper[1, 1])
        self.cx_gripper = self.width_gripper / 2
        self.cy_gripper = self.height_gripper / 2
        self.camera_intrinsics_gripper = CameraIntrinsics(
            frame="camera_gripper",
            fx=self.fx_gripper,
            fy=self.fy_gripper,
            cx=self.cx_gripper,
            cy=self.cy_gripper,
            skew=0.0,
            height=self.height_gripper,
            width=self.width_gripper,
        )

        # Creating instances for calculating suction score and force threshold
        self.suction_score_object_gripper = calcualte_suction_score(
            self.camera_intrinsics_gripper
        )
        self.force_object = calcualte_force()
