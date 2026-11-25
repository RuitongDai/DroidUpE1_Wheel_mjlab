"""Register E1-Wheel velocity tasks with the mjlab CLI."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import e1_wheel_flat_env_cfg, e1_wheel_rough_env_cfg
from .rl_cfg import e1_wheel_ppo_runner_cfg
from .runner import E1WheelOnPolicyRunner

FLAT_TASK_ID = "Mjlab-Velocity-Flat-DroidUp-E1-Wheel"
ROUGH_TASK_ID = "Mjlab-Velocity-Rough-DroidUp-E1-Wheel"

register_mjlab_task(
    task_id=FLAT_TASK_ID,
    env_cfg=e1_wheel_flat_env_cfg(),
    play_env_cfg=e1_wheel_flat_env_cfg(play=True),
    rl_cfg=e1_wheel_ppo_runner_cfg(rough=False),
    runner_cls=E1WheelOnPolicyRunner,
)
register_mjlab_task(
    task_id=ROUGH_TASK_ID,
    env_cfg=e1_wheel_rough_env_cfg(),
    play_env_cfg=e1_wheel_rough_env_cfg(play=True),
    rl_cfg=e1_wheel_ppo_runner_cfg(rough=True),
    runner_cls=E1WheelOnPolicyRunner,
)
