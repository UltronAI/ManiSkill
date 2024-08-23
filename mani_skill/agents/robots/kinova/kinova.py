from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


class KinovaBase(BaseAgent):
    ee_link_name = "end_effector_link"
    
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    
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
        
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
            # pd_ee_delta_pose_align=dict(arm=arm_pd_ee_delta_pose_align),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel),
        )
        
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[0.5, 0.5, -0.5, 0.5]),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ] 
    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class KinovaDoF6(KinovaBase):
    uid = "kinova_dof6"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof6.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0, 0.26, -2.27, 0, -0.61, 1.57,
            ]),
            pose=sapien.Pose(),
        )
    )
    arm_joint_names = [f"joint_{i+1}" for i in range(6)]


@register_agent()
class KinovaDoF7(KinovaBase):
    uid = "kinova_dof7"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof7.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0, 0.26, 3.14, -2.27, 0, -0.61, 1.57,
            ]),
            pose=sapien.Pose(),
        )
    )
    arm_joint_names = [f"joint_{i+1}" for i in range(7)]


