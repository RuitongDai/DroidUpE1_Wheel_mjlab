# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Shared E1-Wheel MDP and flat/rough scene configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.tasks.velocity.mdp import out_of_terrain_bounds, terrain_levels_vel
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from droidup_e1_wheel.assets.e1_wheel.e1_wheel_constants import (
    BASE_LINK,
    JOINT_VELOCITY_LIMITS,
    LEG_JOINT_NAMES,
    LEG_JOINT_TYPES,
    POLICY_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_LINK_NAMES,
    get_e1_wheel_robot_cfg,
)

from .mdp import events, observations, rewards
from .mdp.actions import OrderedJointPositionActionCfg, OrderedJointVelocityActionCfg
from .mdp.commands import E1WheelVelocityCommandCfg
from .terrain_cfg import e1_wheel_rough_terrain_cfg

VELOCITY_COMMAND = "base_velocity"
WHEEL_CONTACT = "wheel_contact"
BODY_CONTACT = "body_contact"
HEIGHT_SCAN = "height_scanner"


def _observation_groups(*, rough: bool, play: bool) -> dict[str, ObservationGroupCfg]:
    wheels = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)
    ordered_joints = SceneEntityCfg(
        "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
    )
    clip = (-100.0, 100.0)
    command = ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": VELOCITY_COMMAND},
        clip=clip,
    )
    action = ObservationTermCfg(func=mdp.last_action, clip=clip)
    actor = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            clip=clip,
            scale=0.25,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            clip=clip,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "velocity_commands": command,
        "joint_pos": ObservationTermCfg(
            func=observations.joint_pos_rel_without_wheel,
            params={"asset_cfg": ordered_joints, "wheel_asset_cfg": wheels},
            clip=clip,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ordered_joints},
            scale=0.05,
            clip=clip,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        ),
        "actions": action,
    }
    # The source critic selects all joints in the articulation's native order.
    critic = {
        "base_lin_vel": ObservationTermCfg(
            func=observations.base_com_lin_vel, clip=clip
        ),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, clip=clip),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, clip=clip),
        "velocity_commands": command,
        "joint_pos": ObservationTermCfg(
            func=observations.joint_pos_rel_without_wheel,
            params={"asset_cfg": SceneEntityCfg("robot"), "wheel_asset_cfg": wheels},
            clip=clip,
        ),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, clip=clip),
        "actions": action,
    }
    if rough:
        critic["height_scan"] = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": HEIGHT_SCAN, "offset": 0.5},
            clip=(-1.0, 1.0),
        )
    return {
        "actor": ObservationGroupCfg(
            terms=actor, concatenate_terms=True, enable_corruption=not play
        ),
        "critic": ObservationGroupCfg(
            terms=critic, concatenate_terms=True, enable_corruption=False
        ),
    }


def _reward_terms() -> dict[str, RewardTermCfg]:
    leg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
    wheel = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)
    front = SceneEntityCfg(
        "robot", joint_names=LEG_JOINT_NAMES[:8], preserve_order=True
    )
    diagonal = SceneEntityCfg(
        "robot",
        joint_names=tuple(
            f"{side}_{kind}_joint" for side in ("RL", "RR") for kind in LEG_JOINT_TYPES
        ),
        preserve_order=True,
    )
    command = {"command_name": VELOCITY_COMMAND}
    return {
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.05),
        "joint_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-2.5e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", actuator_names=list(LEG_JOINT_NAMES)
                )
            },
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": leg}
        ),
        "joint_acc_wheel_l2": RewardTermCfg(
            func=mdp.joint_acc_l2, weight=-2.5e-9, params={"asset_cfg": wheel}
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": leg}
        ),
        "joint_power": RewardTermCfg(
            func=rewards.joint_power, weight=-2e-5, params={"asset_cfg": leg}
        ),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-2.0,
            params={**command, "asset_cfg": leg, "command_threshold": 0.1},
        ),
        "joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-1.0,
            params={
                **command,
                "asset_cfg": leg,
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
            },
        ),
        "wheel_vel_penalty": RewardTermCfg(
            func=rewards.wheel_vel_penalty_without_cmd,
            weight=-0.05,
            params={**command, "asset_cfg": wheel, "command_threshold": 1e-6},
        ),
        "joint_mirror": RewardTermCfg(
            func=rewards.joint_mirror,
            weight=-0.05,
            params={"front_cfg": front, "diagonal_cfg": diagonal},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "undesired_contacts": RewardTermCfg(
            func=rewards.undesired_contacts,
            weight=-1.0,
            params={"sensor_name": BODY_CONTACT, "threshold": 1.0},
        ),
        "contact_forces": RewardTermCfg(
            func=rewards.contact_forces,
            weight=-1.5e-4,
            params={"sensor_name": WHEEL_CONTACT, "threshold": 100.0},
        ),
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=rewards.track_lin_vel_xy_exp,
            weight=3.0,
            params={**command, "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=rewards.track_ang_vel_z_exp, weight=1.5, params={**command, "std": 0.5}
        ),
        "feet_contact_without_cmd": RewardTermCfg(
            func=rewards.feet_contact_without_cmd,
            weight=0.1,
            params={**command, "sensor_name": WHEEL_CONTACT},
        ),
        "upward": RewardTermCfg(func=rewards.upward, weight=1.0),
    }


def _training_events() -> dict[str, EventTermCfg]:
    base = SceneEntityCfg("robot", body_names=(BASE_LINK,))
    return {
        "randomize_contact_material": EventTermCfg(
            func=events.randomize_contact_material,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",))},
        ),
        "randomize_base_mass": EventTermCfg(
            func=events.randomize_mass,
            mode="startup",
            params={"asset_cfg": base, "ranges": (-1.0, 3.0), "additive": True},
        ),
        "randomize_link_mass": EventTermCfg(
            func=events.randomize_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=(".*_(thigh|hip_roll|hip_yaw|calf|wheel)_link",)
                ),
                "ranges": (0.7, 1.3),
                "additive": False,
            },
        ),
        "randomize_base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={"asset_cfg": base, "operation": "add", "ranges": (-0.05, 0.05)},
        ),
        "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.2),
                    "roll": (-3.14, 3.14),
                    "pitch": (-3.14, 3.14),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": dict.fromkeys(
                    ("x", "y", "z", "roll", "pitch", "yaw"), (-0.5, 0.5)
                ),
            },
        ),
        "randomize_external_wrench": EventTermCfg(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": base,
                "force_range": (-10.0, 10.0),
                "torque_range": (-10.0, 10.0),
            },
        ),
        "randomize_actuator_gains": EventTermCfg(
            func=events.randomize_actuator_gains,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }


def _velocity_env_cfg(*, rough: bool, play: bool) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": get_e1_wheel_robot_cfg()}
    cfg.scene.num_envs = 1 if play else 4096
    cfg.scene.terrain = (
        e1_wheel_rough_terrain_cfg(play=play)
        if rough
        else TerrainEntityCfg(terrain_type="plane")
    )
    cfg.scene.sensors = (
        ContactSensorCfg(
            name=WHEEL_CONTACT,
            primary=ContactMatch(mode="body", pattern=WHEEL_LINK_NAMES, entity="robot"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=3,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name=BODY_CONTACT,
            primary=ContactMatch(
                mode="body",
                pattern=("(base_link|.*_(thigh|hip_yaw|calf)_link)",),
                entity="robot",
            ),
            fields=("force",),
            reduce="netforce",
            history_length=3,
        ),
    )
    if rough:
        cfg.scene.sensors += (
            RayCastSensorCfg(
                name=HEIGHT_SCAN,
                frame=ObjRef(type="body", name=BASE_LINK, entity="robot"),
                ray_alignment="yaw",
                pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
                max_distance=5.0,
                include_geom_groups=(0,),
            ),
        )
    wheel_speed_limit = JOINT_VELOCITY_LIMITS["wheel"]
    cfg.actions = {
        "joint_pos": OrderedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=LEG_JOINT_NAMES,
            preserve_order=True,
            use_default_offset=True,
            scale={
                ".*_hip_(roll|yaw)_joint": 0.125,
                "(?!.*_hip_(roll|yaw)_joint).*": 0.25,
            },
            clip={".*": (-100.0, 100.0)},
        ),
        "joint_vel": OrderedJointVelocityActionCfg(
            entity_name="robot",
            actuator_names=WHEEL_JOINT_NAMES,
            preserve_order=True,
            use_default_offset=True,
            scale=5.0,
            clip={".*": (-wheel_speed_limit, wheel_speed_limit)},
        ),
    }
    cfg.commands = {
        VELOCITY_COMMAND: E1WheelVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.05,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=E1WheelVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.5, 1.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }
    cfg.observations = _observation_groups(rough=rough, play=play)
    cfg.rewards = _reward_terms()
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)
    }
    if rough:
        cfg.terminations["terrain_out_of_bounds"] = TerminationTermCfg(
            func=out_of_terrain_bounds, time_out=True, params={"margin": 3.0}
        )
    cfg.curriculum = {}
    if rough and not play:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel, params={"command_name": VELOCITY_COMMAND}
        )
    cfg.events = (
        {"reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")}
        if play
        else _training_events()
    )
    cfg.sim.nconmax = 192 if rough else 96
    cfg.viewer.body_name = BASE_LINK
    if play:
        cfg.episode_length_s = 1.0e9
    return cfg


def e1_wheel_flat_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _velocity_env_cfg(rough=False, play=play)


def e1_wheel_rough_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _velocity_env_cfg(rough=True, play=play)
