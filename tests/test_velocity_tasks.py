import subprocess
import sys

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from droidup_e1_wheel.assets.e1_wheel.e1_wheel_constants import (
    JOINT_VELOCITY_LIMITS,
    LEG_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
)
from droidup_e1_wheel.tasks.velocity import FLAT_TASK_ID, ROUGH_TASK_ID
from droidup_e1_wheel.tasks.velocity.env_cfgs import (
    BODY_CONTACT,
    VELOCITY_COMMAND,
    WHEEL_CONTACT,
)
from droidup_e1_wheel.tasks.velocity.mdp.rewards import wheel_vel_penalty_without_cmd


def test_cli_discovers_external_tasks():
    result = subprocess.run(
        [sys.executable, "-m", "mjlab.scripts.list_envs", "--keyword", "E1-Wheel"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert FLAT_TASK_ID in result.stdout
    assert ROUGH_TASK_ID in result.stdout


@pytest.mark.parametrize(
    ("task_id", "terrain", "iterations"),
    [(FLAT_TASK_ID, "plane", 5000), (ROUGH_TASK_ID, "generator", 20_000)],
)
def test_task_configs(task_id, terrain, iterations):
    assert task_id in list_tasks()
    train = load_env_cfg(task_id)
    play = load_env_cfg(task_id, play=True)
    rough = terrain == "generator"

    assert (
        train.scene.terrain.terrain_type == play.scene.terrain.terrain_type == terrain
    )
    assert train.scene.num_envs == 4096 and play.scene.num_envs == 1
    assert ("terrain_levels" in train.curriculum) == rough
    assert not play.curriculum
    assert ("height_scan" in train.observations["critic"].terms) == rough
    assert "height_scan" not in train.observations["actor"].terms
    assert "base_lin_vel" not in train.observations["actor"].terms
    assert tuple(train.observations["actor"].terms) == (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    assert tuple(train.actions) == ("joint_pos", "joint_vel")
    assert train.actions["joint_pos"].actuator_names == LEG_JOINT_NAMES
    assert train.actions["joint_vel"].actuator_names == WHEEL_JOINT_NAMES
    for cfg in (train, play):
        assert cfg.sim.mujoco.timestep == 0.0025
        assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
        for sensor in cfg.scene.sensors:
            if sensor.name in (BODY_CONTACT, WHEEL_CONTACT):
                assert sensor.history_length * cfg.sim.mujoco.timestep == pytest.approx(
                    0.015
                )
    assert load_rl_cfg(task_id).max_iterations == iterations


@pytest.mark.parametrize(
    ("task_id", "critic_dim"), [(FLAT_TASK_ID, 72), (ROUGH_TASK_ID, 259)]
)
def test_reset_step_and_wheel_io_on_cpu(task_id, critic_dim):
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 2
    cfg.seed = 11
    if cfg.scene.terrain.terrain_generator is not None:
        # A short terrain ladder keeps the CPU test small.
        cfg.scene.terrain.terrain_generator.num_rows = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (2, 69)
        assert obs["critic"].shape == (2, critic_dim)
        assert env.action_manager.get_term("joint_pos").target_names == list(
            LEG_JOINT_NAMES
        )
        assert env.action_manager.get_term("joint_vel").target_names == list(
            WHEEL_JOINT_NAMES
        )
        position_obs = env.observation_manager.get_term_cfg("actor", "joint_pos")
        assert position_obs.params["asset_cfg"].joint_names == list(POLICY_JOINT_NAMES)

        actions = torch.zeros(2, 20)
        actions[:, -4:] = torch.tensor([0.1, 0.2, -0.3, -3.0])
        for _ in range(4):
            obs, reward, terminated, _, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert not terminated.any()

        robot = env.scene["robot"]
        wheel_ids = env.action_manager.get_term("joint_vel").target_ids
        wheel_limit = JOINT_VELOCITY_LIMITS["wheel"]
        torch.testing.assert_close(
            robot.data.joint_vel_target[:, wheel_ids],
            (actions[:, -4:] * 5.0).clamp(-wheel_limit, wheel_limit),
        )
        relative_pos = position_obs.func(env, **position_obs.params)
        torch.testing.assert_close(relative_pos[:, -4:], torch.zeros(2, 4))

        command = env.command_manager.get_term(VELOCITY_COMMAND)
        command.vel_command_b[0] = torch.tensor([0.0, 0.0, 0.0])
        command.vel_command_b[1] = torch.tensor([0.0, 0.0, 0.2])
        params = cfg.rewards["wheel_vel_penalty"].params
        penalty = wheel_vel_penalty_without_cmd(env, **params)
        assert penalty[0] > 0.0
        assert penalty[1] == 0.0
    finally:
        env.close()
