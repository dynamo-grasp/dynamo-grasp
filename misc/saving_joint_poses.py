import torch

joint_poses = torch.tensor(
    [
        [0.0644, -1.8688, 1.1600, 1.0955, 1.4822, -0.2578],
        [0.4511, -1.9977, 1.3855, 0.5800, 2.0622, 2.3844],
    ]
)

torch.save(joint_poses, "misc/joint_poses.pt")
joints = torch.load("misc/joint_poses.pt")
