import argparse

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry import rotation_conversions as rot_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.motion_helper import generate_trajectory, compute_delta_pose, visualize_trajectory


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="TestBench-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run.")
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="human")
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
        
    # target_pose = Pose.create_from_pq(p, q)
    start_pose = env.agent.controller.controllers['arm'].ee_pose_at_base

    target_EE_p = env.cube.pose.p + torch.tensor([0, -0.4, 0.05])
    target_EE_q = rot_utils.rpy_to_quaternion(
                torch.tensor([-np.pi/2, np.pi, 0.0])).repeat(target_EE_p.shape[0], 1)  # (n, 4)
    target_EE_pose = Pose.create_from_pq(p=target_EE_p, q=target_EE_q)
    base_pose = env.agent.controller.controllers["arm"].articulation.pose
    target_pose = base_pose.inv() * target_EE_pose

    traj = generate_trajectory([start_pose, target_pose], 10)
    # visualize_trajectory(traj, "pose_trajectory.avi")

    # while True:
    #     pass
    
    count = 0

    while True:
        current_ee_pose = env.agent.controller.controllers["arm"].ee_pose_at_base
        delta_pose = compute_delta_pose(current_ee_pose, traj[count])

        if delta_pose.p.norm() < 1e-3 and rot_utils.quaternion_to_axis_angle(delta_pose.q).norm() < 1e-3:
            print("!!!!! Move to the next pose: #", count)
            print("Target: ", traj[count], traj[count].p, rot_utils.quaternion_to_axis_angle(traj[count].q))
            print("current_ee_pose: ", current_ee_pose, current_ee_pose.p, rot_utils.quaternion_to_axis_angle(current_ee_pose.q))

            count += 1
            delta_pose = compute_delta_pose(current_ee_pose, traj[count])
            
        action = torch.cat([delta_pose.p, rot_utils.quaternion_to_axis_angle(delta_pose.q)], dim=-1)
        
        env.step(action)
        
        if args.render_mode is not None:
            env.render()

        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
            
    # env.close()
    while True:
        env.render()

    if record_dir:
        print(f"Saving video to {record_dir}")
        


if __name__ == "__main__":
    main(parse_args())
