# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class E1WheelVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        planar = self.vel_command_b[env_ids, :2]
        self.vel_command_b[env_ids, :2] = planar * (
            torch.linalg.vector_norm(planar, dim=1, keepdim=True) > 0.2
        )


@dataclass(kw_only=True)
class E1WheelVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> E1WheelVelocityCommand:
        return E1WheelVelocityCommand(self, env)
