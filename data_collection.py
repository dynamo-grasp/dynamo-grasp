#!/usr/bin/env python
import isaacgym
import isaacgymenvs
import torch

import click
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def get_data_path():
    from datetime import datetime
    import random
    import string

    datetime_string = datetime.now().isoformat().replace(":", "")[:-7]
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(6))
    temp_path = (
        f"scenario_grasp_configurations/{datetime_string}-{random_string}-grasp_data/"
    )
    return os.path.expanduser(temp_path)


@click.command()
@click.option("--num-envs", default=1)
def generate(num_envs):
    envs = isaacgymenvs.make(
        seed=0,
        task="UR16eManipulation",
        num_envs=num_envs,
        sim_device="cuda:0",
        rl_device="cuda:0",
        multi_gpu=True,
        headless=False,
        graphics_device_id=0,
        data_path=get_data_path(),
    )
    obs = envs.reset()

    try:
        while True:
            action = torch.tensor(num_envs * [[0.1, 0, 0, 0, 0, 0, 1]])
            obs, reward, done, info = envs.step(action)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    generate()
