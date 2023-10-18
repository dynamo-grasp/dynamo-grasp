from isaacgym import gymtorch
import torch
from isaacgym import gymapi


class GetSensorValues:
    """
    This method is used to get the sensor values (force and camera images) from the gym environment
    """

    def __init__(self) -> None:
        pass

    def refresh_real_time_sensors(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

    def get_force_sensor_values(self, env_count):
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        fsdata = gymtorch.wrap_tensor(_fsdata)
        return -fsdata[env_count][2].detach().cpu().numpy()

    def get_segmask(self, env_count, camera_id):
        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_count],
            self.camera_handles[env_count][camera_id],
            gymapi.IMAGE_SEGMENTATION,
        )
        torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
        return torch_mask_tensor.to(self.device)

    def get_rgb_image(self, env_count, camera_id):
        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_count],
            self.camera_handles[env_count][camera_id],
            gymapi.IMAGE_COLOR,
        )
        torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
        rgb_image_unflattened = torch_rgb_tensor.to(self.device)
        return torch.reshape(
            rgb_image_unflattened, (rgb_image_unflattened.shape[0], -1, 4)
        )[..., :3]

    def get_depth_image(self, env_count, camera_id):
        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[env_count],
            self.camera_handles[env_count][camera_id],
            gymapi.IMAGE_DEPTH,
        )
        torch_depth_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
        return -torch_depth_tensor.to(self.device)
