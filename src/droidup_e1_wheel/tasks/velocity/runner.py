"""Export a saved E1-Wheel policy when training exits."""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner


class E1WheelOnPolicyRunner(MjlabOnPolicyRunner):
    """Export ONNX once per training exit, using a complete checkpoint."""

    _last_checkpoint: Path | None = None

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint = Path(path)
        with TemporaryDirectory(
            prefix=".checkpoint-", dir=checkpoint.parent
        ) as directory:
            staged = Path(directory) / checkpoint.name
            upload_model = self.cfg["upload_model"]
            try:
                self.cfg["upload_model"] = False
                super().save(str(staged), infos)
            finally:
                self.cfg["upload_model"] = upload_model
            # Preserve the previous checkpoint if saving is interrupted.
            staged.replace(checkpoint)
            self._last_checkpoint = checkpoint
        if upload_model:
            self.logger.save_model(path, self.current_learning_iteration)

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        except (KeyboardInterrupt, Exception) as training_error:
            try:
                self._export_last_checkpoint()
            except Exception as export_error:  # noqa: BLE001
                training_error.add_note(f"E1-Wheel ONNX export failed: {export_error}")
            raise
        else:
            self._export_last_checkpoint()

    def _export_last_checkpoint(self) -> None:
        if self.gpu_global_rank != 0 or self.logger.log_dir is None:
            return
        checkpoint = self._last_checkpoint
        if checkpoint is None:
            print(
                "[INFO] ONNX export skipped because no checkpoint was saved.",
                flush=True,
            )
            return

        # The export wrapper is a copy. A partially updated live actor does not
        # affect the exported policy, and loading its weights leaves training state alone.
        model = self.alg.get_policy().as_onnx(verbose=False).cpu().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "actor_state_dict"
        ]
        model.load_state_dict({name: state[name] for name in model.state_dict()})

        output = Path(self.logger.log_dir) / "policy.onnx"
        with TemporaryDirectory(prefix=".onnx-", dir=output.parent) as directory:
            staged = Path(directory) / output.name
            torch.onnx.export(
                model,
                model.get_dummy_inputs(),
                str(staged),
                input_names=model.input_names,
                output_names=model.output_names,
                opset_version=18,
                dynamo=False,
                external_data=False,
            )
            graph = onnx.load(staged, load_external_data=False)
            if any(
                onnx.external_data_helper.uses_external_data(tensor)
                for tensor in graph.graph.initializer
            ):
                raise RuntimeError("The exported policy references external weights.")
            onnx.checker.check_model(graph)
            staged.replace(output)
        print(f"[INFO] Exported {checkpoint.name} to {output}", flush=True)
