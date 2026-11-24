# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-specific randomization for E1-Wheel."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from torch import Tensor


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_mass(
    env: ManagerBasedRlEnv,
    env_ids: Tensor | None,
    asset_cfg: SceneEntityCfg,
    ranges: tuple[float, float],
    additive: bool,
) -> None:
    ids = resolve_env_ids(env, env_ids)
    robot = env.scene[asset_cfg.name]
    body_ids = robot.indexing.body_ids[asset_cfg.body_ids]
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    samples = torch.empty((len(ids), len(body_ids)), device=env.device).uniform_(
        *ranges
    )
    mass = default_mass + samples if additive else default_mass * samples
    env.sim.model.body_mass[ids[:, None], body_ids] = mass
    env.sim.model.body_inertia[ids[:, None], body_ids] = default_inertia * (
        mass / default_mass
    ).unsqueeze(-1)


@requires_model_fields("geom_friction", "geom_solref")
def randomize_contact_material(
    env: ManagerBasedRlEnv, env_ids: Tensor | None, asset_cfg: SceneEntityCfg
) -> None:
    """Sample effective friction and estimate restitution via contact damping."""
    ids = resolve_env_ids(env, env_ids)
    geom_ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
    buckets = torch.randint(64, (len(ids), len(geom_ids)), device=env.device)
    friction = torch.empty(64, device=env.device).uniform_(0.3, 0.8)
    restitution = torch.empty(64, device=env.device).uniform_(0.0, 0.5)
    log_restitution = restitution.clamp_min(1.0e-6).log()
    damping_ratio = -log_restitution / (math.pi**2 + log_restitution.square()).sqrt()
    env.sim.model.geom_friction[ids[:, None], geom_ids, 0] = friction[buckets]
    env.sim.model.geom_solref[ids[:, None], geom_ids, 1] = damping_ratio[buckets]


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_actuator_gains(
    env: ManagerBasedRlEnv, env_ids: Tensor | None, asset_cfg: SceneEntityCfg
) -> None:
    ids = resolve_env_ids(env, env_ids)
    ctrl_ids = env.scene[asset_cfg.name].indexing.ctrl_ids[asset_cfg.actuator_ids]
    biases = env.sim.get_default_field("actuator_biasprm")[ctrl_ids]
    kp_scale, kd_scale = torch.empty(
        (2, len(ids), len(ctrl_ids)), device=env.device
    ).uniform_(0.5, 2.0)
    kp = -biases[:, 1] * kp_scale
    kd = -biases[:, 2] * kd_scale
    # Position actuators have a position bias and use kp on their input.
    # Velocity actuators have no position bias and use kd on their input.
    gain = torch.where(biases[:, 1] == 0.0, kd, kp)
    env.sim.model.actuator_gainprm[ids[:, None], ctrl_ids, 0] = gain
    env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 1] = -kp
    env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 2] = -kd
