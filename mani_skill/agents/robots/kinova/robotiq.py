from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


def add_gripper_constraint(robot, scene):
    outer_knuckle = robot.active_joints_map["right_outer_knuckle_joint"]
    outer_finger  = robot.active_joints_map["right_inner_finger_joint"]
    inner_knuckle = robot.active_joints_map["right_inner_knuckle_joint"]
    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (
        outer_finger.get_global_pose().p
        + inner_knuckle.get_global_pose().p
        - outer_knuckle.get_global_pose().p
    )
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = torch.matmul(T_fw[..., :3, :3], p_w[..., None]).squeeze(-1) + T_fw[..., :3, 3] # (B, 3)
    p_p = torch.matmul(T_pw[..., :3, :3], p_w[..., None]).squeeze(-1) + T_pw[..., :3, 3] # (B, 3)
    right_drive = scene.create_drive(lif, Pose(p_f), pad, Pose(p_p))
    right_drive.set_limit_x(low=0, high=0, stiffness=0, damping=0)
    right_drive.set_limit_y(low=0, high=0, stiffness=0, damping=0)
    right_drive.set_limit_z(low=0, high=0, stiffness=0, damping=0)

    outer_knuckle = robot.active_joints_map["finger_joint"]
    outer_finger  = robot.active_joints_map["left_inner_finger_joint"]
    inner_knuckle = robot.active_joints_map["left_inner_knuckle_joint"]
    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (
        outer_finger.get_global_pose().p
        + inner_knuckle.get_global_pose().p
        - outer_knuckle.get_global_pose().p
    )
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = torch.matmul(T_fw[..., :3, :3], p_w[..., None]).squeeze(-1) + T_fw[..., :3, 3] # (B, 3)
    p_p = torch.matmul(T_pw[..., :3, :3], p_w[..., None]).squeeze(-1) + T_pw[..., :3, 3] # (B, 3)

    left_drive = scene.create_drive(lif, Pose(p_f), pad, Pose(p_p))
    left_drive.set_limit_x(low=0, high=0, stiffness=0, damping=0)
    left_drive.set_limit_y(low=0, high=0, stiffness=0, damping=0)
    left_drive.set_limit_z(low=0, high=0, stiffness=0, damping=0)
    
    print("Done adding gripper constraint")


@register_agent()
class Robotiq(BaseAgent):
    uid = "robotiq"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/robotiq_2f_85/robotiq_85.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            finger_joint=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_outer_knuckle=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            ),
            pose=sapien.Pose(),
        )
    )
    
    gripper_joint_names = [
        "right_outer_knuckle_joint",
        # "right_inner_finger_joint",
        # "right_inner_knuckle_joint",
        "finger_joint",
        # "left_inner_finger_joint",
        # "left_inner_knuckle_joint"
    ]
    ee_link_name = "end_effector_link"
    
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 0.1
    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,  # a trick to have force when the object is thin
            upper=0.8,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )
        
        controller_configs = dict(
            pd_joint_pos=dict(gripper=gripper_pd_joint_pos),
        )
        
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def _after_init(self):
        add_gripper_constraint(self.robot, self.scene)
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold