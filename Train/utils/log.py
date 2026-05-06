from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


class Logger:
    def __init__(
        self,
        log_path: str | None = None,
        run_name: str | None = None,
        log_file: str | None = None,
        tensorboard_logdir: str | None = None,
    ):
        self.tensorboard_logdir = None
        self.tensorboard_writer = None
        if log_file is None:
            if run_name is None and log_path and log_path.endswith(".txt"):
                log_file = log_path
            elif log_path is not None and run_name is not None:
                run_dir_name = self._build_run_dir_name(run_name)
                self.run_dir_name = run_dir_name
                self.run_dir = os.path.join(log_path, run_dir_name)
                self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
                self.final_model_dir = os.path.join(self.checkpoint_dir, "model-final")
                log_file = os.path.join(self.run_dir, "log.txt")
            else:
                raise ValueError(
                    "Either log_file or both log_path and run_name must be set."
                )
        else:
            self.run_dir = os.path.dirname(log_file) or "."
            self.run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
            self.final_model_dir = os.path.join(self.checkpoint_dir, "model-final")

        self.log_file = log_file
        if not hasattr(self, "run_dir"):
            self.run_dir = os.path.dirname(self.log_file) or "."
            self.run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
            self.final_model_dir = os.path.join(self.checkpoint_dir, "model-final")
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if tensorboard_logdir is not None:
            self._init_tensorboard(tensorboard_logdir)

    def _build_run_dir_name(self, run_name: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_run_name = re.sub(r"[\\/:\s]+", "_", run_name.strip())
        safe_run_name = safe_run_name.strip("_") or "run"
        return f"{timestamp}_{safe_run_name}"

    def onlylog(self, message: Any, name: str | None = None) -> None:
        with open(self.log_file, "a") as f:
            f.write(self.format_message(message, name) + "\n")

    def logandprint(self, message: Any, name: str | None = None) -> None:
        formatted = self.format_message(message, name)
        print(formatted)
        self.onlylog(formatted)

    def log_tensorboard(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        if self.tensorboard_writer is None:
            return

        if step is None:
            step = self._to_tensorboard_step(metrics.get("global_step"))

        for key, value in metrics.items():
            if key == "global_step":
                continue
            scalar = self._to_tensorboard_scalar(value)
            if scalar is None:
                continue
            self.tensorboard_writer.add_scalar(
                self._build_tensorboard_tag(key, prefix),
                scalar,
                step,
            )
        self.tensorboard_writer.flush()

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None

    def format_message(self, message: Any, name: str | None = None) -> str:
        if isinstance(message, Mapping):
            return self._format_titled_block(message, name or "dict")
        if self._is_sequence(message):
            return self._format_titled_block(message, name or "list")
        if name:
            return f"{name}: {self._format_scalar(message)}"
        return self._format_scalar(message)

    def _format_titled_block(self, value: Any, title: str) -> str:
        return "\n".join([title, *self._format_block(value, indent_level=1)])

    def _format_block(self, value: Any, indent_level: int) -> list[str]:
        if isinstance(value, Mapping):
            return self._format_mapping(value, indent_level)
        if self._is_sequence(value):
            return self._format_sequence(value, indent_level)
        return [f"{self._indent(indent_level)}{self._format_scalar(value)}"]

    def _format_mapping(
        self,
        mapping: Mapping[Any, Any],
        indent_level: int,
    ) -> list[str]:
        lines = []
        indent = self._indent(indent_level)
        for key, value in mapping.items():
            if isinstance(value, Mapping) or self._is_sequence(value):
                lines.append(f"{indent}{key}:")
                lines.extend(self._format_block(value, indent_level + 1))
            else:
                lines.append(f"{indent}{key}: {self._format_scalar(value)}")
        return lines

    def _format_sequence(
        self,
        sequence: Sequence[Any],
        indent_level: int,
    ) -> list[str]:
        lines = []
        indent = self._indent(indent_level)
        for value in sequence:
            if isinstance(value, Mapping) or self._is_sequence(value):
                lines.append(f"{indent}-")
                lines.extend(self._format_block(value, indent_level + 1))
            else:
                lines.append(f"{indent}- {self._format_scalar(value)}")
        return lines

    def _format_scalar(self, value: Any) -> str:
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (TypeError, ValueError, RuntimeError):
                pass
        return str(value)

    def _is_sequence(self, value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        )

    def _indent(self, indent_level: int) -> str:
        return "  " * indent_level

    def _init_tensorboard(self, tensorboard_logdir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires PyTorch and tensorboard to be installed."
            ) from exc

        self.tensorboard_logdir = os.path.join(
            tensorboard_logdir,
            getattr(self, "run_dir_name", "run"),
        )
        os.makedirs(self.tensorboard_logdir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_logdir)

    def _build_tensorboard_tag(self, key: str, prefix: str | None) -> str:
        tag = str(key)
        if prefix:
            metric_prefix = f"{prefix}_"
            if tag.startswith(metric_prefix):
                tag = tag[len(metric_prefix) :]
            return f"{prefix}/{tag}"
        return tag

    def _to_tensorboard_scalar(self, value: Any) -> float | None:
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (TypeError, ValueError, RuntimeError):
                return None
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        return float(value)

    def _to_tensorboard_step(self, value: Any) -> int | None:
        scalar = self._to_tensorboard_scalar(value)
        if scalar is None:
            return None
        return int(scalar)
