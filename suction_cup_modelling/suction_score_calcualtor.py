import math
import torch
from autolab_core import DepthImage
from homogeneous_transformation_and_conversion.rotation_conversions import *


class calcualte_suction_score:
    def __init__(self, camera_intrinsics):
        self.camera_intrinsics = camera_intrinsics
        self.suction_projection_bound = torch.tensor([0.0375, 0.0175])
        self.num_suction_projections = 8
        self.flexion_deofrmation_thresh = 0.008
        self.device = "cuda:0"

    def convert_depth_to_point_cloud(self):
        camera_u = torch.arange(0, self.camera_intrinsics.width, device=self.device)
        camera_v = torch.arange(0, self.camera_intrinsics.height, device=self.device)
        camera_v, camera_u = torch.meshgrid(camera_v, camera_u, indexing="ij")

        Z = self.depth_image
        X = (camera_u - self.camera_intrinsics.cx) / self.camera_intrinsics.fx * Z
        Y = (camera_v - self.camera_intrinsics.cy) / self.camera_intrinsics.fy * Z

        depth_bar = 10
        Z = Z.view(-1)
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)

        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[
            :, valid
        ]
        position = position.permute(1, 0)
        points = position[:, 0:3]
        return points

    def convert_uv_point_to_xyz_point(self, u, v):
        Z = self.depth_image[v, u]
        X = (u - self.camera_intrinsics.cx) / self.camera_intrinsics.fx * Z
        Y = (v - self.camera_intrinsics.cy) / self.camera_intrinsics.fy * Z

        depth_bar = 10
        Z = Z.view(-1)
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)

        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[
            :, valid
        ]
        position = position.permute(1, 0)
        points = position[:, 0:3]
        if len(points) == 0:
            return torch.tensor([0, 0, -1]).to(self.device)
        return points[0].to(self.device)

    def convert_xyz_point_to_uv_point(self, xyz_point):
        X = xyz_point[:, 0]
        Y = xyz_point[:, 1]
        Z = xyz_point[:, 2]
        U = X * self.camera_intrinsics.fx / Z + self.camera_intrinsics.cx
        V = Y * self.camera_intrinsics.fy / Z + self.camera_intrinsics.cy

        return U, V

    def find_nearest(self, centroid, points):
        """
        Identify the closest point in `suction points projection` to each suction point with respect to camera and return them.
        """
        suction_points = centroid[:2] + self.suction_coordinates[:, :2]
        distances = torch.cdist(
            points[:, :2].type(torch.float64), suction_points.type(torch.float64)
        )
        min_indices = torch.argmin(distances, dim=0)
        point_cloud_suction = points[min_indices]
        return point_cloud_suction

    def calculator(
        self, depth_image, segmask, grasps_and_predictions, object_id, offset
    ):
        """
        Compute suction score and pose adjustments for robotic arm during suction-based manipulation.

        Parameters:
        - depth_image, segmask: Input image data for processing.
        - grasps_and_predictions: sample points obtained from dexnet method.
        - object_id: Identifier for the target object.
        - offset: Offset for compensating the cropping of the image.

        Returns:
        Tuple containing:
        - suction_score: Computed score indicating suction effectiveness.
        - Pre-grasp Pose: The configuration of the robot arm prior to executing a pushing action.
        - centroid_angle: Angle of the object's normal vector.
        """
        self.depth_image = depth_image
        self.segmask = segmask
        self.grasps_and_predictions = grasps_and_predictions
        self.object_id = object_id
        self.pre_grasp_pose_x = torch.min(depth_image)

        # Calculate object normals
        self.depth_image_normal = depth_image.clone().cpu().numpy()
        depth_normal = DepthImage(
            self.depth_image_normal, frame=self.camera_intrinsics.frame
        )
        point_cloud_im = self.camera_intrinsics.deproject_to_image(depth_normal)
        self.normal_cloud_im = point_cloud_im.normal_cloud_im()

        self.segmask = self.segmask == self.object_id
        self.depth_image[self.segmask == 0] = 0
        centroid_angle = torch.tensor([0, 0, 0]).to(self.device)

        # Refine segmask and calculate centroid using point cloud median
        points = self.convert_depth_to_point_cloud()
        centroid_point = torch.FloatTensor(
            [
                torch.median(points[:, 0]),
                torch.median(points[:, 1]),
                torch.median(points[:, 2]),
            ]
        ).to(self.device)
        if centroid_point.any() == float("nan"):
            return (
                torch.tensor(0),
                torch.tensor([0, 0, 0]),
                torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]]),
            )

        # Obtain xyz coordinates, handle cases with negative z-values, and initialize suction_coordinates
        if self.grasps_and_predictions == None:
            xyz_point = self.convert_uv_point_to_xyz_point(
                int(self.segmask.shape[1] / 2), int(self.segmask.shape[0] / 2)
            )
        else:
            xyz_point = self.convert_uv_point_to_xyz_point(
                self.grasps_and_predictions.center.x + offset[0],
                self.grasps_and_predictions.center.y + offset[1],
            )
        if xyz_point[2] < 0:
            return (
                torch.tensor(0),
                torch.tensor([0, 0, 0]),
                torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]]),
            )

        # Iterate through all the projections to calculate and append suction coordinates
        # 0.02 m is the radius of the suction cup
        base_coordinate = torch.tensor([0.02, 0, 0]).to(self.device)
        self.suction_coordinates = base_coordinate.view(1, 3)
        for angle in range(45, 360, 45):
            x = base_coordinate[0] * math.cos(angle * math.pi / 180) - base_coordinate[
                1
            ] * math.sin(angle * math.pi / 180)
            y = base_coordinate[0] * math.sin(angle * math.pi / 180) + base_coordinate[
                1
            ] * math.cos(angle * math.pi / 180)
            self.suction_coordinates = torch.cat(
                (self.suction_coordinates, torch.tensor([[x, y, 0.0]]).to(self.device)),
                dim=0,
            ).type(torch.float64)
        if self.grasps_and_predictions != None:
            centroid_angle = torch.tensor(
                self.normal_cloud_im.data[
                    int(self.grasps_and_predictions.center.y + offset[1])
                ][int(self.grasps_and_predictions.center.x + offset[0])]
            ).to(self.device)
            centroid_angle = torch.tensor(
                [0.0, -centroid_angle[1], centroid_angle[0]]
            ).to(self.device, dtype=torch.float64)
            self.suction_coordinates = torch.mm(
                self.suction_coordinates,
                euler_angles_to_matrix(
                    centroid_angle.clone().detach().to(self.device),
                    "XYZ",
                    degrees=False,
                ).type(torch.float64),
                out=None,
            )
        else:
            centroid_angle = torch.tensor(
                self.normal_cloud_im[int(self.camera_intrinsics.height / 2)][
                    int(self.camera_intrinsics.width / 2)
                ]
            ).to(self.device)
            centroid_angle = torch.tensor(
                [0.0, -centroid_angle[1], centroid_angle[0]]
            ).to(self.device, dtype=torch.float64)
        # Calcualting the suction projection points with respect to the base coordinate (camera)
        point_cloud_suction = self.find_nearest(xyz_point, points)
        U, V = self.convert_xyz_point_to_uv_point(point_cloud_suction)

        for i in range(len(self.suction_coordinates)):
            if self.segmask[V[i].type(torch.int)][U[i].type(torch.int)] == 0:
                if grasps_and_predictions == None:
                    return (torch.tensor(0), torch.tensor([0, 0, 0]), centroid_angle)
                else:
                    return (
                        torch.tensor(0),
                        torch.tensor(
                            [self.pre_grasp_pose_x, -xyz_point[0], -xyz_point[1]]
                        ),
                        centroid_angle,
                    )

        # Evaluate difference in xy plane between obtained point cloud and suction coordinates projections
        difference_xy_plane = point_cloud_suction[:, :2] - (
            self.suction_coordinates[:, :2] + xyz_point[:2]
        )
        thresh = torch.sum(torch.sum(difference_xy_plane, 1))

        if abs(thresh) > self.flexion_deofrmation_thresh:
            if grasps_and_predictions == None:
                return (
                    torch.tensor(0),
                    torch.tensor([0, 0, 0]),
                    centroid_angle,
                )
            else:
                return (
                    torch.tensor(0),
                    torch.tensor([self.pre_grasp_pose_x, -xyz_point[0], -xyz_point[1]]),
                    centroid_angle,
                )

        # This block of code is related to calculating the suction score and
        # transforming the robotic arm to ensure the end-effector (eef) is facing
        # the normal of the target object. While this code is kept commented and
        # not in active use, it might be valuable for future reference or alternative
        # approaches.

        # The following section calcualtes point_cloud_suction_facing_normal. This
        # ensures that the eef faces towards the normal derived from the centroid angle.
        #
        # point_cloud_suction_facing_normal = torch.mm(
        #     point_cloud_suction.clone().detach().type(torch.float64),
        #     euler_angles_to_matrix(
        #         torch.tensor([centroid_angle[1], -centroid_angle[2], 0.0]).to(
        #             self.device
        #         ),
        #         "XYZ",
        #         degrees=False,
        #     ).type(torch.float64),
        #     out=None,
        # )
        # minimum_suction_point = torch.min(point_cloud_suction_facing_normal[:, 2]).to(
        #     self.device
        # )

        # ri calculates a normalized relative inverse distance score, clamped to [0,1],
        # representing the inverted and scaled proximity of each point to the minimal suction point.
        #
        # ri = torch.clamp(
        #     torch.abs(point_cloud_suction_facing_normal[:, 2] - minimum_suction_point)
        #     / 0.023,
        #     max=1.0,
        # )

        # suction_score_facing_normal calculates the final suction score as 1 minus
        # the maximum value in ri, providing a measure of suction efficacy based on proximity.
        #
        # suction_score_facing_normal = 1 - torch.max(ri)

        # Calcualte the conical spring suction deformation score
        # 0.023 m is the height of the suction cup
        minimum_suction_point = torch.min(point_cloud_suction[:, 2]).to(self.device)
        ri = torch.clamp(
            torch.abs(point_cloud_suction[:, 2] - minimum_suction_point) / 0.023,
            max=1.0,
        )
        suction_score = 1 - torch.max(ri)

        # Return calculated suction_score along with pre grasp pose and angle of the centre of suction cup
        if grasps_and_predictions == None:
            return (
                suction_score,
                torch.tensor([xyz_point[2] - 0.07, -xyz_point[0], -xyz_point[1]]),
                torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]]),
            )
        return (
            suction_score,
            torch.tensor([self.pre_grasp_pose_x, -xyz_point[0], -xyz_point[1]]),
            torch.tensor([0, 0, 0]),
        )

    def calculate_contact(self, depth_image, segmask, object_id):
        """
        Calculate the existence of a valid contact between the suction points and the object surface.

        Parameters:
        - depth_image (tensor): Depth image of the scene.
        - segmask (tensor): Segmentation mask identifying objects.
        - object_id (int): Identifier of the target object.

        Returns:
        - torch.tensor: Binary indicator (0/1) of whether a valid contact is made.
        """
        self.depth_image = depth_image
        self.segmask = segmask
        self.object_id = object_id

        self.depth_image[self.segmask != self.object_id] = 0
        self.depth_image = torch.nan_to_num(self.depth_image, nan=0)
        # Centroid method using median of point cloud
        points = self.convert_depth_to_point_cloud()
        centroid_point = torch.FloatTensor(
            [
                torch.median(points[:, 0]),
                torch.median(points[:, 1]),
                torch.median(points[:, 2]),
            ]
        ).to(self.device)
        if centroid_point.any() == float("nan"):
            return torch.tensor(0)
        # Given sample point convert to xyz point
        xyz_point = self.convert_uv_point_to_xyz_point(
            int(self.segmask.shape[1] / 2), int(self.segmask.shape[0] / 2)
        )

        # Store the base projections of the suction cup points
        base_coordinate = torch.tensor([0.02, 0, 0]).to(self.device)
        self.suction_coordinates = base_coordinate.view(1, 3)
        for angle in range(45, 360, 45):
            x = base_coordinate[0] * math.cos(angle * math.pi / 180) - base_coordinate[
                1
            ] * math.sin(angle * math.pi / 180)
            y = base_coordinate[0] * math.sin(angle * math.pi / 180) + base_coordinate[
                1
            ] * math.cos(angle * math.pi / 180)
            self.suction_coordinates = torch.cat(
                (self.suction_coordinates, torch.tensor([[x, y, 0.0]]).to(self.device)),
                dim=0,
            ).type(torch.float64)
        point_cloud_suction = self.find_nearest(xyz_point, points)
        U, V = self.convert_xyz_point_to_uv_point(point_cloud_suction)

        # Ensure at least 6 points of the suction cup are facing the object
        count = 0
        for i in range(self.num_suction_projections):
            if torch.isnan(U[i]) or torch.isnan(V[i]):
                count += 1
                continue
            elif self.segmask[V[i].type(torch.int)][U[i].type(torch.int)] != object_id:
                count += 1
        if count >= 2:
            return torch.tensor(0)

        # Check depth constraints to ensure viable suction contact
        count = 0
        furthest_suction_projection_point = -1000.0
        nearest_suction_projection_point = 1000.0
        for i in range(self.num_suction_projections):
            if (
                torch.isnan(U[i])
                or torch.isnan(V[i])
                or self.depth_image[V[i].type(torch.int)][U[i].type(torch.int)] == 0
            ):
                continue
            if (
                furthest_suction_projection_point
                < self.depth_image[V[i].type(torch.int)][U[i].type(torch.int)]
            ):
                furthest_suction_projection_point = self.depth_image[
                    V[i].type(torch.int)
                ][U[i].type(torch.int)]
            if (
                nearest_suction_projection_point
                > self.depth_image[V[i].type(torch.int)][U[i].type(torch.int)]
            ):
                nearest_suction_projection_point = self.depth_image[
                    V[i].type(torch.int)
                ][U[i].type(torch.int)]

        if (
            furthest_suction_projection_point > self.suction_projection_bound[0]
            or nearest_suction_projection_point > self.suction_projection_bound[1]
        ):
            return torch.tensor(0)

        # Ensure that at least 6 points have depth within tolerance
        for i in range(self.num_suction_projections):
            if torch.isnan(U[i]) or torch.isnan(V[i]):
                count += 1
                continue
            if (
                furthest_suction_projection_point
                - self.depth_image[V[i].type(torch.int)][U[i].type(torch.int)]
            ) > self.suction_projection_bound[1] or self.depth_image[
                V[i].type(torch.int)
            ][
                U[i].type(torch.int)
            ] > self.suction_projection_bound[
                1
            ]:
                count += 1

        if count >= 2:
            return torch.tensor(0)

        return torch.tensor(1)
