import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from homogeneous_transformation_and_conversion.rotation_conversions import *


class RobotControl:
    def __init__(self) -> None:
        pass

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(self._j_eef, 1, 2)
            @ m_eef
            @ (
                self.kp * dpose[:, :6] - self.kd * self.states["wrist_3_link_vel"]
            ).unsqueeze(-1)
        )

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.ur16e_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, self.num_ur16e_dofs :] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye((6), device=self.device).unsqueeze(0)
            - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv
        ) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(
            u.squeeze(-1),
            -self._ur16e_effort_limits[:6].unsqueeze(0),
            self._ur16e_effort_limits[:6].unsqueeze(0),
        )
        return u

    def deploy_actions(self, env_ids, pos):
        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def execute_control_actions(self):
        self.actions = self.actions.clone().detach().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_temp = self._compute_osc_torques(dpose=u_arm)
            u_arm = torch.clip(u_temp, min=-10, max=10)
        self._arm_control[:, :6] = u_arm

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )

    def calculate_grasp_action(self, env_count):
        # Transformation for grasp pose (wg --> wo*og)
        rotation_matrix_grasp_pose = euler_angles_to_matrix(
            self.grasp_angle[env_count].to(self.device),
            "XYZ",
            degrees=False,
        ).type(torch.float)
        translation_grasp_pose = (
            torch.tensor([self.ee_vel[env_count], 0, 0])
            .to(self.device)
            .type(torch.float)
        )
        translation_grasp_pose = torch.matmul(
            rotation_matrix_grasp_pose, translation_grasp_pose
        )
        # Get the transformation of pre grasp pose, object pose, and end effector pose
        start_point = self.T_world_to_pre_grasp_pose[:3, 3].clone().detach()
        end_point = self.T_world_to_object[:3, 3].clone().detach()
        current_point = self.T_world_to_ee_pose[:3, 3].clone().detach()
        v = torch.tensor(
            [
                end_point[0] - start_point[0],
                end_point[1] - start_point[1],
                end_point[2] - start_point[2],
            ]
        )
        # Calculate the vector connecting p1 to p
        w = torch.tensor(
            [
                current_point[0] - start_point[0],
                current_point[1] - start_point[1],
                current_point[2] - start_point[2],
            ]
        )
        # Calculate the projection of w onto v
        t = (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]) / (
            v[0] ** 2 + v[1] ** 2 + v[2] ** 2
        )
        # Calculate the closest point on the line to p
        q = torch.tensor(
            [
                start_point[0] + t * v[0],
                start_point[1] + t * v[1],
                start_point[2] + t * v[2],
            ]
        ).to(self.device)
        # Find the distance between p and q
        distance = current_point - q
        self.action_env = torch.tensor(
            [
                [
                    self.ee_vel[env_count],
                    translation_grasp_pose[1]
                    - distance[1] * 100 * self.ee_vel[env_count],
                    translation_grasp_pose[2]
                    - distance[2] * 100 * self.ee_vel[env_count],
                    self.ee_vel[env_count] * 100 * self.action_orientation[0],
                    self.ee_vel[env_count] * 100 * self.action_orientation[1],
                    self.ee_vel[env_count] * 100 * self.action_orientation[2],
                    1,
                ]
            ],
            dtype=torch.float,
        )
