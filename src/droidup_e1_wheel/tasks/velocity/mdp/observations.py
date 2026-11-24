# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from torch import Tensor


def base_com_lin_vel(env: ManagerBasedRlEnv) -> Tensor:
    return env.scene["robot"].data.root_com_lin_vel_b


def joint_pos_rel_without_wheel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, wheel_asset_cfg: SceneEntityCfg
) -> Tensor:
    """Keep all position channels and mask out the continuous wheel angles."""
    robot = env.scene[asset_cfg.name]
    relative_pos = robot.data.joint_pos - robot.data.default_joint_pos
    relative_pos[:, wheel_asset_cfg.joint_ids] = 0.0
    return relative_pos[:, asset_cfg.joint_ids]
