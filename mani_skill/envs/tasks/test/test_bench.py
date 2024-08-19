from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Panda, PandaStick, PandaWristCam, Xmate3Robotiq, Kinova, KinovaRobotiq, Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TestBench-v1", max_episode_steps=1000)
class TestBench(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "kinova_robotiq", "kinova", "robotiq"]
    agent: Union[Panda, Xmate3Robotiq, KinovaRobotiq, Kinova, Robotiq]
    
    def __init__(self, *args, robot_uids="robotiq", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        
        head_cam_pose = sapien_utils.look_at(eye=[-0.615, 0, 0.5, ], target=[0.1, 0, 0])
        head_cam = CameraConfig("head_camera",
                                pose=head_cam_pose,
                                width=128,
                                height=128,
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
        self.cube_half_size = 0.05
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
        )
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = -0.4
            # ensure cube is spawned on the left side of the table
            xyz[:, 1] = 0.0
            xyz[:, 2] = self.cube_half_size
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz))

    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 
    