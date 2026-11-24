# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward terms whose definitions differ from the mjlab velocity task."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import generated_commands, joint_pos_rel
from mjlab.managers import SceneEntityCfg
from torch import Tensor


def _upright_factor(env: ManagerBasedRlEnv) -> Tensor:
    gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    return (-gravity_z).clamp(0.0, 0.7) / 0.7


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> Tensor:
    target = generated_commands(env, command_name)[:, :2]
    actual = env.scene["robot"].data.root_com_lin_vel_b[:, :2]
    error = (target - actual).square().sum(dim=1)
    return torch.exp(-error / std**2) * _upright_factor(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> Tensor:
    target = generated_commands(env, command_name)[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-(target - actual).square() / std**2) * _upright_factor(env)


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> Tensor:
    return env.scene["robot"].data.root_com_lin_vel_b[:, 2].square()


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> Tensor:
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def joint_power(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return (
        (robot.data.joint_vel[:, ids] * robot.data.qfrc_actuator[:, ids])
        .abs()
        .sum(dim=1)
    )


def stand_still(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float,
) -> Tensor:
    deviation = joint_pos_rel(env, asset_cfg=asset_cfg).abs().sum(dim=1)
    command_speed = torch.linalg.vector_norm(
        generated_commands(env, command_name), dim=1
    )
    return deviation * (command_speed < command_threshold) * _upright_factor(env)


def joint_pos_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> Tensor:
    deviation = torch.linalg.vector_norm(joint_pos_rel(env, asset_cfg=asset_cfg), dim=1)
    command_speed = torch.linalg.vector_norm(
        generated_commands(env, command_name), dim=1
    )
    body_speed = torch.linalg.vector_norm(
        env.scene["robot"].data.root_com_lin_vel_b[:, :2], dim=1
    )
    moving = (command_speed > command_threshold) | (body_speed > velocity_threshold)
    return torch.where(
        moving, deviation, deviation * stand_still_scale
    ) * _upright_factor(env)


def wheel_vel_penalty_without_cmd(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float,
) -> Tensor:
    command_speed = torch.linalg.vector_norm(
        generated_commands(env, command_name), dim=1
    )
    wheel_speed = (
        env.scene[asset_cfg.name]
        .data.joint_vel[:, asset_cfg.joint_ids]
        .abs()
        .sum(dim=1)
    )
    return wheel_speed * (command_speed <= command_threshold)


def joint_mirror(
    env: ManagerBasedRlEnv, front_cfg: SceneEntityCfg, diagonal_cfg: SceneEntityCfg
) -> Tensor:
    positions = env.scene[front_cfg.name].data.joint_pos
    error = (
        positions[:, front_cfg.joint_ids] - positions[:, diagonal_cfg.joint_ids]
    ).square()
    return 0.5 * error.sum(dim=1) * _upright_factor(env)


def _peak_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> Tensor:
    history = env.scene[sensor_name].data.force_history
    return torch.linalg.vector_norm(history, dim=-1).amax(dim=-1)


def undesired_contacts(
    env: ManagerBasedRlEnv, sensor_name: str, threshold: float
) -> Tensor:
    return (_peak_contact_force(env, sensor_name) > threshold).sum(dim=1)


def contact_forces(
    env: ManagerBasedRlEnv, sensor_name: str, threshold: float
) -> Tensor:
    return (_peak_contact_force(env, sensor_name) - threshold).clamp_min(0.0).sum(dim=1)


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> Tensor:
    first_contacts = env.scene[sensor_name].compute_first_contact(env.step_dt)
    command_speed = torch.linalg.vector_norm(
        generated_commands(env, command_name), dim=1
    )
    return first_contacts.sum(dim=1) * (command_speed < 0.1) * _upright_factor(env)


def upward(env: ManagerBasedRlEnv) -> Tensor:
    return (1.0 - env.scene["robot"].data.projected_gravity_b[:, 2]).square()
