import argparse

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry import rotation_conversions as rot_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers import RecordEpisode


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run.")
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("-p", "--pause", action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is no seed",
    )
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main(args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if args.seed is not None:
        np.random.seed(args.seed)
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        shader_dir=args.shader,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        parallel_in_single_scene=parallel_in_single_scene,
        **args.env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=env._max_episode_steps)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    if args.render_mode is not None:
        viewer = env.render()
        viewer.paused = args.pause
        env.render()
        
    p = torch.tensor([0.3, -0.3, 0.0])
    q = rot_utils.axis_angle_to_quaternion(torch.tensor([0, 0, torch.pi/2]))
    target_pose = Pose.create_from_pq(p, q)
    start_pose = env.agent.controller.controllers['arm'].ee_pose_at_base
    traj = generate_trajectory([start_pose, target_pose], [10], env.agent.controller.control_freq)
    
    print("===== initial states =====")
    print("EE pose @ world frame", env.agent.controller.controllers['arm'].ee_pose)
    print("Base pose @ world frame", env.agent.robot.get_links()[0].pose)
    print("EE pose @ base frame", env.agent.controller.controllers['arm'].ee_pose_at_base)
    print("Target pose @ base frame", target_pose)
    print("Length of trajectory:", len(traj))
    print("==========================")
    
    
    for i in range(len(traj)-1):
        delta_pose = compute_delta_pose(traj[i], traj[i+1])
        action = torch.cat([delta_pose[0], delta_pose[1]], dim=1)
        
        print(">>")
        print("Previous EE pose @ world frame", env.agent.controller.controllers['arm'].ee_pose)
        print("Previous EE pose @ base frame", env.agent.controller.controllers['arm'].ee_pose_at_base)
        
        _ = env.step(action)
        
        print("Action:", action)
        print("Expect pose:", traj[i+1].p, traj[i+1].q)
        
        print("EE pose @ world frame", env.agent.controller.controllers['arm'].ee_pose)
        print("EE pose @ base frmae:", env.agent.controller.controllers['arm'].ee_pose_at_base)
        
        
        if args.render_mode is not None:
            env.render()

        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
            
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")
        

def interpolate_pose(pose1, pose2, alpha: float):
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


def generate_trajectory(poses, durations, control_frequency):
    """
    Generates a smooth trajectory to control the end-effector.
    
    :param poses: List of poses, where each pose is [x, y, z, qw, qx, qy, qz]
    :param durations: List of durations between consecutive poses
    :param control_frequency: Scalar frequency to determine the number of internal poses per second
    :return: List of interpolated poses (smooth trajectory)
    """
    trajectory = []
    num_segments = len(poses) - 1

    for i in range(num_segments):
        start_pose = poses[i]
        end_pose = poses[i + 1]
        duration = durations[i]
        
        num_steps = int(duration * control_frequency)
        for step in range(num_steps):
            alpha = step / num_steps
            interp_pose = interpolate_pose(start_pose, end_pose, alpha)
            trajectory.append(interp_pose)
    
    # Add the last pose to ensure the trajectory ends exactly at the final pose
    trajectory.append(poses[-1])
    
    return trajectory


def compute_delta_pose(pose1: Pose, pose2: Pose) -> (torch.Tensor, torch.Tensor):
    """
    Computes the delta position and delta rotation (in Euler angles) between two poses.
    
    :param pose1: The initial Pose object.
    :param pose2: The final Pose object.
    :return: A tuple containing delta position (torch.Tensor) and delta rotation (in Euler angles, torch.Tensor).
    """
    # Delta Position
    delta_position = pose2.p - pose1.p
    
    # Compute relative rotation (delta rotation)
    quat1 = pose1.q
    quat2 = pose2.q
    
    # Calculate the relative quaternion (pose1 to pose2)
    relative_quat = quaternion_multiply(quat2, quaternion_conjugate(quat1))
    
    # Convert the relative quaternion to Euler angles
    relative_rotation = rot_utils.quaternion_to_axis_angle(relative_quat)
    delta_rotation = torch.tensor(relative_rotation, device=pose1.device)
    
    return delta_position, delta_rotation

def quaternion_conjugate(quat):
    """
    Computes the conjugate of a quaternion.
    
    :param quat: The quaternion tensor [w, x, y, z].
    :return: The conjugated quaternion tensor.
    """
    conjugate = quat.clone()
    conjugate[..., 1:] *= -1  # Negate the vector part
    return conjugate

def quaternion_multiply(quat1, quat2):
    """
    Multiplies two quaternions.
    
    :param quat1: First quaternion tensor [w, x, y, z].
    :param quat2: Second quaternion tensor [w, x, y, z].
    :return: Resulting quaternion after multiplication.
    """
    w1, x1, y1, z1 = quat1[..., 0], quat1[..., 1], quat1[..., 2], quat1[..., 3]
    w2, x2, y2, z2 = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack((w, x, y, z), dim=-1)


if __name__ == "__main__":
    main(parse_args())
