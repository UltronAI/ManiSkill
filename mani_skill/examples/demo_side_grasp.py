import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry import rotation_conversions as rot_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.motion_helper import generate_trajectory #, compute_delta_pose, visualize_trajectory
import time

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--demo", type=str, default="grasp_bottle", 
                        choices=["grasp_bottle", "open_door"],
                        help="The demo you want to run")
    parser.add_argument("-e", "--env-id", type=str, default="TestBench-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-r", "--robot-uids", type=str, default="kinova_dof6", 
                        choices=["panda", "kinova_dof7", "kinova_dof6"],
                        help="Robot setups supported are ['panda']")
    parser.add_argument("-i", "--init-mode", type=str, default="up", 
                        choices=["up", "side", "forward"],
                        help="The initial pose of the robot")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run.")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose", 
                        help="The control mode to use for the robot")
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-video", action="store_true", default=False, help="Record video of the simulation")
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


def compute_delta_pose(pose1, pose2):
    delta_p = pose2.p - pose1.p
    # import pdb; pdb.set_trace()
    delta_q = rot_utils.quaternion_multiply(pose1.q, pose2.q)
    delta_euler = rot_utils.quaternion_to_euler_angles(delta_q, "XYZ")
    print("delta_p:", delta_p)
    print("delta_euler:", delta_euler)
    return torch.cat([delta_p, delta_euler], dim=-1)


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
    
    # euler_offset = torch.tensor([0.0, 0.0, 0.0])
    if args.robot_uids == "panda":
        robot_uids = "panda_wristcam"
        euler_offset = torch.tensor([0.0, 0.0, np.pi/2])
    elif args.robot_uids == "kinova_dof7":
        robot_uids = "kinova_dof7_robotiq_2f85"
    elif args.robot_uids == "kinova_dof6":
        robot_uids = "kinova_dof6_robotiq_2f85"
    else:
        raise ValueError(f"Unknown robot_uids: {args.robot_uid}")
        
    env: BaseEnv = gym.make(
        args.env_id,
        init_mode=args.init_mode,
        robot_uids=robot_uids,
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
    if args.record_video:
        record_dir = "videos/{}-{}-{}-{}".format(args.env_id, args.demo, args.init_mode, robot_uids)
        os.makedirs(record_dir, exist_ok=True)
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
        
    if "panda" in robot_uids:
        offset = np.pi/2
    else:
        offset = 0.0
    
    pose0 = env.agent.controller.controllers["arm"].ee_pose
    keyposes = [pose0]
    # ============================
    if "panda" in robot_uids:
        if args.demo == "grasp_bottle":
            # Demo: grasp bottle
            # keypoint 1:
            cube_pos = env.cube.pose.p
            pos1 = cube_pos + torch.tensor([0.0, 0.0, 0.5])
            quat1 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, 0.0, 0.0]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            # import pdb; pdb.set_trace()
            pose1 = Pose.create_from_pq(p=pos1, q=quat1)
            # keypoint 2:
            pos2 = cube_pos + torch.tensor([0.0, -0.5, 0.05])
            quat2 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi/2]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose2 = Pose.create_from_pq(p=pos2, q=quat2)
            # keypoint 3:
            pos3 = cube_pos + torch.tensor([0.0, -0.02, 0.05])
            quat3 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi/2]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose3 = Pose.create_from_pq(p=pos3, q=quat3)
            # keypoint 4:
            pos4 = cube_pos + torch.tensor([0.0, -0.02, 0.5])
            quat4 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi/2]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose4 = Pose.create_from_pq(p=pos4, q=quat4)
            keyposes += [pose2, pose3, pose4]
        elif args.demo == "open_door":
            # Demo: open door
            door_pos = env.cube.pose.p + torch.tensor([0.1, 0.0, 0.4])
            # keypoint 1:
            pos1 = door_pos + torch.tensor([0.0, 0.1, 0.0])
            quat1 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/2]), "XYZ").repeat(args.num_envs, 1)
            pose1 = Pose.create_from_pq(p=pos1, q=quat1)
            # keypoint 2:
            pos2 = door_pos + torch.tensor([0.0, 0.1 * np.cos(np.pi/6), -0.1 * np.sin(np.pi/6)])
            quat2 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/2+np.pi/6]), "XYZ").repeat(args.num_envs, 1)
            pose2 = Pose.create_from_pq(p=pos2, q=quat2)
            # keypoint 3:
            pos3 = door_pos + torch.tensor([0.0, 0.1 * np.cos(np.pi/3), -0.1 * np.sin(np.pi/3)])
            quat3 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/2+np.pi/3]), "XYZ").repeat(args.num_envs, 1)
            pose3 = Pose.create_from_pq(p=pos3, q=quat3)
            # keypoint 3:
            pos4 = door_pos + torch.tensor([0.0, 0.0, -0.1])
            quat4 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/2+np.pi/2]), "XYZ").repeat(args.num_envs, 1)
            pose4 = Pose.create_from_pq(p=pos4, q=quat4)
            keyposes += [pose1, pose2]
        else:
            raise ValueError(f"Unknown demo: {args.demo}")
    else:
        if args.demo == "grasp_bottle":
            # Demo: grasp bottle
            # keypoint 1:
            cube_pos = env.cube.pose.p
            pos1 = cube_pos + torch.tensor([0.0, 0.0, 0.5])
            quat1 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, 0.0, -np.pi/2]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            # import pdb; pdb.set_trace()
            pose1 = Pose.create_from_pq(p=pos1, q=quat1)
            # keypoint 2:
            pos2 = cube_pos + torch.tensor([0.0, -0.5, 0.05])
            quat2 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose2 = Pose.create_from_pq(p=pos2, q=quat2)
            # keypoint 3:
            pos3 = cube_pos + torch.tensor([0.0, -0.02, 0.05])
            quat3 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose3 = Pose.create_from_pq(p=pos3, q=quat3)
            # keypoint 4:
            pos4 = cube_pos + torch.tensor([0.0, -0.02, 0.5])
            quat4 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([-np.pi/2, 0.0, -np.pi]), "XYZ").repeat(args.num_envs, 1)  # (n, 4)
            pose4 = Pose.create_from_pq(p=pos4, q=quat4)
            keyposes += [pose2, pose3, pose4]
        elif args.demo == "open_door":
            # Demo: open door
            door_pos = env.cube.pose.p + torch.tensor([0.1, 0.0, 0.4])
            # keypoint 1:
            pos1 = door_pos + torch.tensor([0.0, 0.1, 0.0])
            quat1 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, 0.0]), "XYZ").repeat(args.num_envs, 1)
            pose1 = Pose.create_from_pq(p=pos1, q=quat1)
            # keypoint 2:
            pos2 = door_pos + torch.tensor([0.0, 0.1 * np.cos(np.pi/6), -0.1 * np.sin(np.pi/6)])
            quat2 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/6]), "XYZ").repeat(args.num_envs, 1)
            pose2 = Pose.create_from_pq(p=pos2, q=quat2)
            # keypoint 3:
            pos3 = door_pos + torch.tensor([0.0, 0.1 * np.cos(np.pi/3), -0.1 * np.sin(np.pi/3)])
            quat3 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/3]), "XYZ").repeat(args.num_envs, 1)
            pose3 = Pose.create_from_pq(p=pos3, q=quat3)
            # keypoint 3:
            pos4 = door_pos + torch.tensor([0.0, 0.0, -0.1])
            quat4 = rot_utils.euler_angles_to_quaternion(
                torch.tensor([np.pi, np.pi/2, np.pi/2]), "XYZ").repeat(args.num_envs, 1)
            pose4 = Pose.create_from_pq(p=pos4, q=quat4)
            keyposes += [pose1, pose2]
        else:
            raise ValueError(f"Unknown demo: {args.demo}")
    
    trajectory = generate_trajectory(keyposes, 5)
    
    count = 0
    
    current_ee_pose = env.agent.controller.controllers["arm"].ee_pose
    delta_pose = compute_delta_pose(current_ee_pose, trajectory[count])
    delta_action = torch.cat([delta_pose, torch.zeros([args.num_envs, 1])], dim=-1)
    env.step(delta_action)

    while True:
        print("Towards pose #", count, trajectory[count], "...")
        current_ee_pose = env.agent.controller.controllers["arm"].ee_pose
        delta_pose = compute_delta_pose(current_ee_pose, trajectory[count])
        delta_action = torch.cat([delta_pose, torch.zeros([args.num_envs, 1])], dim=-1)

        if delta_pose.norm() < 1e-1:
            print("!!!!! Move to the next pose: #", count)
            count += 1
            if count >= len(trajectory):
                break
            delta_pose = compute_delta_pose(current_ee_pose, trajectory[count])
            delta_action = torch.cat([delta_pose, torch.zeros([args.num_envs, 1])], dim=-1)
            # env.step(delta_action)
            
        obs, reward, terminated, truncated, info = env.step(delta_action)
        # env.scene.step()
        
        if args.render_mode is not None:
            env.render()
            
        time.sleep(0.01)
        
    env.close()

    if args.record_video:
        print(f"Saving video to {record_dir}")
        


if __name__ == "__main__":
    main(parse_args())
