from types import SimpleNamespace

import onnx
import pytest
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from onnx.reference import ReferenceEvaluator
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from droidup_e1_wheel.tasks.velocity.runner import E1WheelOnPolicyRunner


@pytest.mark.parametrize(
    "error_type",
    [None, KeyboardInterrupt, RuntimeError],
    ids=["finished", "ctrl-c", "failure"],
)
def test_training_exit_exports_saved_actor(tmp_path, monkeypatch, error_type):
    observations = TensorDict({"actor": torch.zeros(1, 3)}, batch_size=[1])
    actor = MLPModel(observations, {"actor": ["actor"]}, "actor", 2, hidden_dims=(4,))
    runner = E1WheelOnPolicyRunner.__new__(E1WheelOnPolicyRunner)
    runner.alg = SimpleNamespace(
        save=lambda: {"actor_state_dict": actor.state_dict()},
        get_policy=lambda: actor,
    )
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=42))
    runner.cfg = {"upload_model": False}
    runner.logger = SimpleNamespace(log_dir=str(tmp_path))
    runner.gpu_global_rank = 0
    runner.current_learning_iteration = 0
    completed = tmp_path / "model_0.pt"
    policy = tmp_path / "policy.onnx"
    with torch.inference_mode():
        expected = actor(observations)

    def learn_then_exit(self, num_learning_iterations, init_at_random_ep_len):
        self.save(str(completed))
        assert not policy.exists()
        if error_type is not None:
            with torch.no_grad():
                actor.mlp[-1].bias.add_(10)
            raise error_type("training stopped")

    monkeypatch.setattr(MjlabOnPolicyRunner, "learn", learn_then_exit)
    if error_type is None:
        runner.learn(1)
    else:
        with pytest.raises(error_type, match="training stopped"):
            runner.learn(1)
        with torch.inference_mode():
            torch.testing.assert_close(actor(observations), expected + 10)

    assert {path.name for path in tmp_path.iterdir()} == {"model_0.pt", "policy.onnx"}
    model = onnx.load(policy, load_external_data=False)
    (result,) = ReferenceEvaluator(model).run(
        None, {"obs": observations["actor"].numpy()}
    )
    torch.testing.assert_close(torch.from_numpy(result), expected)
