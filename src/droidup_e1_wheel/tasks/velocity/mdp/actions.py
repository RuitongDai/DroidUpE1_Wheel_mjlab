"""Joint actions that preserve the policy order on mjlab 1.6.0."""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import (
    BaseActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names


def _order_targets(
    targets: tuple[list[int], list[str]], cfg: BaseActionCfg
) -> tuple[list[int], list[str]]:
    if not cfg.preserve_order:
        return targets
    ids, names = targets
    order, ordered_names = resolve_matching_names(
        cfg.actuator_names, names, preserve_order=True
    )
    return [ids[index] for index in order], ordered_names


class OrderedJointPositionAction(JointPositionAction):
    def _find_targets(self, cfg: BaseActionCfg) -> tuple[list[int], list[str]]:
        return _order_targets(super()._find_targets(cfg), cfg)


class OrderedJointVelocityAction(JointVelocityAction):
    def _find_targets(self, cfg: BaseActionCfg) -> tuple[list[int], list[str]]:
        return _order_targets(super()._find_targets(cfg), cfg)


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env: ManagerBasedRlEnv) -> OrderedJointPositionAction:
        return OrderedJointPositionAction(self, env)


@dataclass(kw_only=True)
class OrderedJointVelocityActionCfg(JointVelocityActionCfg):
    def build(self, env: ManagerBasedRlEnv) -> OrderedJointVelocityAction:
        return OrderedJointVelocityAction(self, env)
