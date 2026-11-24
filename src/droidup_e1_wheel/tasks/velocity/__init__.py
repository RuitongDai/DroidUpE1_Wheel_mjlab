"""Register E1-Wheel velocity tasks with the mjlab CLI."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import e1_wheel_flat_env_cfg, e1_wheel_rough_env_cfg
from .rl_cfg import e1_wheel_ppo_runner_cfg

FLAT_TASK_ID = "Mjlab-Velocity-Flat-DroidUp-E1-Wheel"
ROUGH_TASK_ID = "Mjlab-Velocity-Rough-DroidUp-E1-Wheel"

register_mjlab_task(
    task_id=FLAT_TASK_ID,
    env_cfg=e1_wheel_flat_env_cfg(),
    play_env_cfg=e1_wheel_flat_env_cfg(play=True),
    rl_cfg=e1_wheel_ppo_runner_cfg(rough=False),
)
register_mjlab_task(
    task_id=ROUGH_TASK_ID,
    env_cfg=e1_wheel_rough_env_cfg(),
    play_env_cfg=e1_wheel_rough_env_cfg(play=True),
    rl_cfg=e1_wheel_ppo_runner_cfg(rough=True),
)
