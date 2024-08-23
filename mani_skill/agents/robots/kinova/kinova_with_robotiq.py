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
from .kinova import KinovaDoF6, KinovaDoF7

"""
EEF frame is LEFT-UP-FORWARD
"""

@register_agent()
class KinovaDoF7Robotiq2f85(KinovaDoF7):
    uid = "kinova_dof7_robotiq_2f85"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof7_robotiq_2f_85.urdf"
    # disable_self_collisions = True
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    keyframes = dict(
        rest=Keyframe(
            qpos=[0, 0.26, 3.14, -2.27, 0, -0.61, 1.57, # arm
                  0, 0, 0, 0, 0, 0], # gripper
            pose=sapien.Pose()
        )
    )
    # Robotiq Gripper
    finger_joint_names = [
        "left_outer_knuckle_joint", 
        "right_outer_knuckle_joint"
    ]
    passive_finger_joint_names = [
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]

    gripper_stiffness = 1e5 
    gripper_damping = 1e3 
    gripper_force_limit = 0.1
    gripper_friction = 0.05
    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        controller_configs = super()._controller_configs

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joints = PassiveControllerConfig(
            joint_names=self.passive_finger_joint_names,
            damping=0,
            friction=0,
        )
        # use a mimic controller config to define one action to control both fingers
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.finger_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=self.finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
        )
        
        for key in controller_configs.keys():
            if 'delta' in key:
                controller_configs[key].update(dict(
                    finger=finger_mimic_pd_joint_delta_pos,
                    passive_finger_joints=passive_finger_joints,
                ))
            else:
                controller_configs[key].update(dict(
                    finger=finger_mimic_pd_joint_pos,
                    passive_finger_joints=passive_finger_joints,
                ))
        
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)


@register_agent()
class KinovaDoF7Robotiq2f140(KinovaDoF7):
    uid = "kinova_dof7_robotiq_2f140"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof7_robotiq_2f_140.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=[0, 0.26, 3.14, -2.27, 0, -0.61, 1.57, # arm
                  0, 0, 0, 0, 0, 0], # gripper
            pose=sapien.Pose()
        )
    )


@register_agent()
class KinovaDoF6Robotiq2f85(KinovaDoF6):
    uid = "kinova_dof6_robotiq_2f85"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof6_robotiq_2f_85.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=[0, 0.26, -2.27, 0, -0.61, 1.57, # arm
                  0, 0, 0, 0, 0, 0], # gripper
            pose=sapien.Pose()
        )
    )


@register_agent()
class KinovaDoF6Robotiq2f140(KinovaDoF6):
    uid = "kinova_dof6_robotiq_2f140"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/kinova/gen3_dof6_robotiq_2f_140.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=[0, 0.26, -2.27, 0, -0.61, 1.57, # arm
                  0, 0, 0, 0, 0, 0], # gripper
            pose=sapien.Pose()
        )
    )

