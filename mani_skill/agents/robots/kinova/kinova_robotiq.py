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
    outer_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "right_outer_knuckle_joint"
    )
    outer_finger = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_finger_joint"
    )
    inner_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_knuckle_joint"
    )
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

    outer_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "left_outer_knuckle_joint"
    )
    outer_finger = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_finger_joint"
    )
    inner_knuckle = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_knuckle_joint"
    )
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
class KinovaRobotiq(BaseAgent):
    uid = "kinova_robotiq"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof7_robotiq_2f_85.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_outer_knuckle=dict(
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
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            ),
            pose=sapien.Pose(),
        )
    )
    
    arm_joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ]
    gripper_joint_names = [
        # "right_outer_knuckle_joint",
        # "right_inner_finger_joint",
        # "right_inner_knuckle_joint",
        "left_outer_knuckle_joint",
        # "left_inner_finger_joint",
        # "left_inner_knuckle_joint"
    ]
    ee_link_name = "end_effector_link"
    
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    
    gripper_stiffness = 1e4
    gripper_damping = 2000
    gripper_force_limit = 0.1
    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True
        
        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        
        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )
        
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            # pd_ee_delta_pose_align=dict(arm=arm_pd_ee_delta_pose_align),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos),
        )
        
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def _after_init(self):
        add_gripper_constraint(self.robot, self.scene)
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold