# DroidUp E1-Wheel in mjlab

DroidUp E1-Wheel 的平地和崎岖地形速度跟踪任务，从 `robot_lab` 的 E1-Wheel 配置迁移到 mjlab 1.6.0。

| 任务 ID | 地形 | PPO 默认迭代数 |
| --- | --- | ---: |
| `Mjlab-Velocity-Flat-DroidUp-E1-Wheel` | 平地 | 5,000 |
| `Mjlab-Velocity-Rough-DroidUp-E1-Wheel` | 楼梯、反向楼梯、方块、随机起伏、正反坡 | 20,000 |

## Conda 环境

在仓库根目录创建名为 `e1_wheel` 的 Conda 环境，再用该环境的 pip 安装项目和测试依赖。训练默认使用 NVIDIA CUDA。

```bash
conda create -n e1_wheel python=3.13 -y
conda activate e1_wheel
python -m pip install -e '.[test]'
python -m mjlab.scripts.list_envs --keyword E1-Wheel
```

如果 `e1_wheel` 已经创建但还没有 Python，先用 `conda install --name e1_wheel python=3.13 pip`，再从激活环境开始执行。

如果 Conda 在下载 `.zst` 索引时遇到 SSL 断开，可为该次命令设置 `CONDA_REPODATA_USE_ZST=false`，并加上 `--solver classic`。

## 训练和回放

```bash
train Mjlab-Velocity-Flat-DroidUp-E1-Wheel --env.scene.num-envs 256
train Mjlab-Velocity-Rough-DroidUp-E1-Wheel --env.scene.num-envs 256

play Mjlab-Velocity-Flat-DroidUp-E1-Wheel --agent zero --num-envs 1 --viewer viser
play Mjlab-Velocity-Flat-DroidUp-E1-Wheel --checkpoint-file /path/to/model.pt --num-envs 1 --viewer viser
```

回放配置使用默认站立姿态、关闭训练随机化，并提供 Viser 速度命令控件。训练产物默认写入 `logs/rsl_rl/e1_wheel_flat/` 或 `logs/rsl_rl/e1_wheel_rough/`。测试可在 CPU 上运行。

训练正常结束、按 Ctrl+C 或遇到 Python 异常时，runner 会用最近一次成功保存的 `model_*.pt` 在本次运行目录导出单个 `policy.onnx`，并打印对应的 checkpoint 文件名。定期保存仍只生成 `.pt`，因此中断时导出的策略可能早于最后一轮日志。首次 checkpoint 保存前退出时会提示没有可导出的模型。

```bash
python -m pytest
```

## 迁移细节

- `assets/e1_wheel/` 中的 MJCF 和网格来自 `robot_lab` 的 `E1_DOG_wheel_no_head.urdf`（源提交 `0f46197`）。`e1_wheel_constants.py` 管理机器人初态、关节名称和执行器配置。
- `tasks/velocity/` 管理 Flat/Rough、MDP 和 PPO。动作顺序是 FR、FL、RR、RL 的 16 个腿关节位置目标，再按相同腿序排列 4 个轮关节速度目标。两项任务的 actor 都有 69 维观测，Rough 的高度扫描只给 critic。
- MuJoCo 物理子步为 2.5 ms，每 8 个子步更新一次策略动作，控制频率为 50 Hz。轮地与机身接触力奖励使用最近 15 ms 的力历史。
- MuJoCo 使用单一切向摩擦系数，并以接触阻尼近似随机恢复系数。接触奖励使用 MuJoCo 的逐刚体净接触力历史。轮速目标限幅为 ±13.09 rad/s，原始腿关节速度限制保留为模型参数参考，实际关节速度由 MuJoCo 动力学决定。
- Rough 使用 mjlab 原生地形生成和课程，保留源任务的六类地形和采样比例。网格布局、碰撞和求解结果会随仿真后端变化。建议使用在本项目中生成的 RSL-RL checkpoint。
