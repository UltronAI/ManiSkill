from typing import List

import argparse

import gymnasium as gym
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mani_skill.utils.geometry import rotation_conversions as rot_utils
from mani_skill.utils.structs.pose import Pose


def compute_delta_pose(pose1: Pose, pose2: Pose):
    T1 = pose1.to_transformation_matrix()
    T2 = pose2.to_transformation_matrix()

    delta_T = torch.matmul(T2, torch.inverse(T1))
    
    delta_q = rot_utils.matrix_to_quaternion(delta_T[:, :3, :3])
    delta_p = delta_T[:, :3, 3]

    return Pose.create_from_pq(p=delta_p, q=delta_q)


def interpolate_pose(pose1: Pose, pose2: Pose, alpha: float):
    """
    Interpolates between two Pose objects given an interpolation factor alpha (0 <= alpha <= 1).
    
    :param pose1: The starting Pose object.
    :param pose2: The ending Pose object.
    :param alpha: Interpolation factor (0 = pose1, 1 = pose2).
    :return: A new Pose object that is the interpolated result.
    """
    # Linearly interpolate positions
    interp_position = (1 - alpha) * pose1.p + alpha * pose2.p
    
    # Perform SLERP for quaternion interpolation
    dot_product = torch.sum(pose1.q * pose2.q, dim=-1, keepdim=True)
    if torch.all(dot_product < 0):
        pose2.q = -pose2.q
        dot_product = -dot_product

    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)

    if torch.all(sin_theta_0 > 1e-6):
        sin_theta = torch.sin(alpha * theta_0)
        sin_theta_1 = torch.sin((1 - alpha) * theta_0)
        interp_quat = (sin_theta_1 / sin_theta_0) * pose1.q + (sin_theta / sin_theta_0) * pose2.q
    else:
        interp_quat = pose1.q

    # Normalize the resulting quaternion to avoid numerical drift
    interp_quat = torch.nn.functional.normalize(interp_quat)

    # Create and return the new interpolated Pose
    return Pose.create_from_pq(interp_position, interp_quat)


def generate_trajectory(poses: List[Pose], 
                        num_points: int):
    trajectory = []
    num_segments = len(poses) - 1

    pose1 = poses[0]
    for i in range(num_segments):
        pose2 = poses[i + 1]

        for j in range(num_points):
            alpha = j / num_points
            interp_pose = interpolate_pose(pose1, pose2, alpha)
            trajectory.append(interp_pose)

        pose1 = interp_pose

    trajectory.append(poses[-1])

    return trajectory

def visualize_trajectory(trajectory: List[Pose], video_path: str, frame_size=(500, 500), duration=10, fps=30):
    num_frames = int(duration * fps)
    frame_interval = max(1, len(trajectory) // num_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for i in range(0, len(trajectory), frame_interval):
        pose = trajectory[i].sp
        ax.clear()
        # Plot position
        ax.scatter(pose.p[0].item(), pose.p[1].item(), pose.p[2].item(), c='r', marker='o')

        # Plot orientation (as a set of axes)
        R = pose.to_transformation_matrix()[:3, :3]
        origin = pose.p

        ax.quiver(origin[0], origin[1], origin[2], R[0, 0], R[0, 1], R[0, 2], color='r')
        ax.quiver(origin[0], origin[1], origin[2], R[1, 0], R[1, 1], R[1, 2], color='g')
        ax.quiver(origin[0], origin[1], origin[2], R[2, 0], R[2, 1], R[2, 2], color='b')

        # Set plot limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Capture plot as an image and write to video
        plt.draw()
        plt.pause(0.1)

        # Convert plot to an OpenCV image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, frame_size)

        out.write(img)

        if i == 0:
            import pdb; pdb.set_trace()

    out.release()
    # plt.close(fig)


def test_generate_trajectory_visualization():
    # Define poses for the trajectory
    pose1 = Pose.create_from_pq(p=torch.tensor([0.0, 0.0, 0.0]), q=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    pose2 = Pose.create_from_pq(p=torch.tensor([0.5, 0.5, 0.5]), q=torch.tensor([0.707, 0.0, 0.707, 0.0]))
    pose3 = Pose.create_from_pq(p=torch.tensor([1.0, 1.0, 1.0]), q=torch.tensor([0.0, 0.0, 1.0, 0.0]))

    poses = [pose1, pose2, pose3]
    durations = [2.0, 2.0]  # Each segment takes 2 seconds
    control_frequency = 30  # 30 Hz control frequency

    # Generate trajectory
    trajectory = generate_trajectory(poses, durations, control_frequency)
    # import pdb; pdb.set_trace()

    # Visualize and save the trajectory as a video
    video_path = "pose_trajectory.avi"
    visualize_trajectory(trajectory, video_path)


if __name__ == "__main__":

    test_generate_trajectory_visualization()
