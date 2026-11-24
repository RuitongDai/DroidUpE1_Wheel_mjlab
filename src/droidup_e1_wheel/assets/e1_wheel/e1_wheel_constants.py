"""DroidUp E1-Wheel MJCF and actuator configuration."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

E1_WHEEL_XML = Path(__file__).resolve().parent / "xmls" / "e1_wheel.xml"
BASE_LINK = "base_link"
LEG_JOINT_TYPES = ("thigh", "hip_roll", "hip_yaw", "calf")
LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{kind}_joint" for leg in LEG_ORDER for kind in LEG_JOINT_TYPES
)
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEG_ORDER)
WHEEL_LINK_NAMES = tuple(f"{leg}_wheel_link" for leg in LEG_ORDER)
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
JOINT_VELOCITY_LIMITS = {
    "thigh": 13.09,
    "hip_roll": 13.61,
    "hip_yaw": 15.71,
    "calf": 13.09,
    "wheel": 13.09,
}


def get_spec() -> mujoco.MjSpec:
    """Load a fresh spec so independently built scenes can edit it."""
    spec = mujoco.MjSpec.from_file(str(E1_WHEEL_XML))
    # MjLab attaches the spec to the scene, so resolve the mesh directory here.
    spec.meshdir = str(E1_WHEEL_XML.parent.parent / "meshes")
    return spec


def get_e1_wheel_robot_cfg() -> EntityCfg:
    leg_actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f".*_{kind}_joint",),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            frictionloss=0.0,
        )
        for kind, kp, kd, effort in (
            ("thigh", 60.0, 3.0, 60.0),
            ("hip_roll", 50.0, 2.5, 36.0),
            ("hip_yaw", 30.0, 1.5, 15.0),
            ("calf", 60.0, 3.0, 60.0),
        )
    )
    wheels = BuiltinVelocityActuatorCfg(
        target_names_expr=(".*_wheel_joint",),
        damping=0.5,
        effort_limit=60.0,
        frictionloss=0.0,
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.46),
            joint_pos={
                ".*_thigh_joint": 0.5,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_calf_joint": -1.0,
                ".*_wheel_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(*leg_actuators, wheels),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
