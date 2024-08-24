from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import KinovaDoF7Robotiq2f85, KinovaDoF7, PandaWristCam, KinovaDoF6Robotiq2f85
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry import rotation_conversions as rot_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TestBench-v1", max_episode_steps=5000)
class TestBench(BaseEnv):
    
    SUPPORTED_ROBOTS = ["kinova_dof7_robotiq_2f85" "kinova_dof7", "panda_wristcam", "kinova_dof6_robotiq_2f85"]
    agent: Union[KinovaDoF7Robotiq2f85, KinovaDoF7, PandaWristCam]
    
    def __init__(self, *args, robot_uids="kinova_dof7_robotiq_2f85", robot_init_qpos_noise=0.02, init_mode="up", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.init_mode = init_mode
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        base_cam_pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        base_cam = CameraConfig("base_camera",
                                pose=base_cam_pose,
                                width=128,
                                height=128,
                                fov=np.pi / 2,
                                near=0.01,
                                far=100,)
        
        head_cam_pose = sapien_utils.look_at(eye=[-0.615, 0, 0.7], target=[0.1, 0, 0])
        head_cam = CameraConfig("head_camera",
                                pose=head_cam_pose,
                                width=256,
                                height=256,
                                fov=np.pi / 2,
                                near=0.01,
                                far=100,)
        return [base_cam, head_cam]
    
    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", 
            pose=pose, 
            width=512, 
            height=512, 
            fov=1, 
            near=0.01, 
            far=100
        )
        
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube_half_sizes = [0.03, 0.07, 0.05]
        self.cube = actors.build_box(
            self.scene,
            half_sizes=self.cube_half_sizes,
            color=[1, 0, 0, 1],
            name="cube",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = -0.2
            xyz[:, 1] = 0.0
            xyz[:, 2] = self.cube_half_sizes[-1]
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz))

            if not hasattr(self.agent.controller.controllers["arm"], "kinematics"):
                return
            # assert hasattr(self.agent.controller.controllers["arm"], "kinematics"), "Only avaiable for EE controllers"
            
            # if "panda" in self.robot_uids:
            #     euler_offset = rot_utils.euler_angles_to_quaternion(torch.tensor([0.0, 0.0, np.pi/2]))
            # else:
            #     euler_offset = rot_utils.euler_angles_to_quaternion(torch.tensor([0.0, 0.0, 0.0]))

            # if "panda" in self.robot_uids:
            #     raise NotImplementedError("Only available for kinova_dof7_robotiq_2f85 and kinova_dof7")
            # else:
            if "panda" in self.robot_uids:
                offset = np.pi/2
            else:
                offset = 0.0
            if self.init_mode == "up":
                target_EE_p = self.cube.pose.p + torch.tensor([0.0, 0.0, 0.3])
                target_EE_q = rot_utils.euler_angles_to_quaternion(
                    torch.tensor([np.pi, 0.0, -np.pi/2 + offset]), "XYZ").repeat(len(env_idx), 1)  # (n, 4)
            elif self.init_mode == "side":
                target_EE_p = self.cube.pose.p + torch.tensor([0.0, -0.3, 0.05])  # (n, 3)
                target_EE_q = rot_utils.euler_angles_to_quaternion(
                    torch.tensor([-np.pi/2, 0.0, -np.pi + offset]), "XYZ").repeat(len(env_idx), 1)
            elif self.init_mode == "forward":
                target_EE_p = self.cube.pose.p + torch.tensor([0.0, 0.0, 0.3])
                target_EE_q = rot_utils.euler_angles_to_quaternion(
                    torch.tensor([0.0, np.pi/2, np.pi/2 + offset]), "XYZ").repeat(len(env_idx), 1)
            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")
            
            target_EE_pose = Pose.create_from_pq(p=target_EE_p, q=target_EE_q)

            base_pose = self.agent.controller.controllers["arm"].articulation.pose
            print("Initial base pose: ", base_pose)
            world2base = Pose.create(base_pose.raw_pose[env_idx]).inv()
            target_EE_pose_at_base = world2base * target_EE_pose

            q0 = self.agent.controller.controllers["arm"].articulation.get_qpos()[env_idx]

            qpos = self.agent.controller.controllers["arm"].kinematics.compute_ik(
                target_pose=target_EE_pose_at_base,
                q0=q0,
            )  # (n, 6)
            if qpos is None:
                raise ValueError("IK solution not found")
            
            curr_full_qpos = self.agent.robot.get_qpos()
            if curr_full_qpos.shape[-1] > qpos.shape[-1]:
                rest_qpos = torch.zeros((b, curr_full_qpos.shape[-1] - qpos.shape[-1]))
                qpos = torch.cat([qpos, rest_qpos], dim=-1)
            self.agent.reset(init_qpos=qpos)


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 
