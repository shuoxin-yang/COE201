from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


_MODEL_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?B)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_TARGET_MODULE_ALIASES = {
    "q": ("q_proj", "query_proj", "query", "q"),
    "k": ("k_proj", "key_proj", "key", "k"),
    "v": ("v_proj", "value_proj", "value", "v"),
    "o": ("o_proj", "out_proj", "output_proj", "output", "o"),
}


def build_lora_run_core_name(config: Mapping[str, Any]) -> str:
    global_cfg = _require_mapping(config, "global")
    lora_cfg = _require_mapping(config, "LoRA")

    parts = [
        _extract_model_size(global_cfg.get("model_name")),
        f"R-{_format_number(_require_value(lora_cfg, 'rank'))}",
    ]

    scaling_type = _normalize_scaling_type(lora_cfg.get("scaling_type", "r/a"))
    if scaling_type != "r/a":
        parts.append(f"AL-{_sanitize_name_part(scaling_type, fallback='scaling')}")

    dropout_rate = _to_float(lora_cfg.get("dropout_rate", 0.0), "LoRA.dropout_rate")
    if dropout_rate != 0:
        parts.append(f"DR-{_format_number(dropout_rate)}")

    parts.append(_build_layer_selection_name(lora_cfg))
    parts.append(_build_target_module_name(_require_value(lora_cfg, "target_modules")))
    return "_".join(parts)


def build_lora_run_dir_name(
    config: Mapping[str, Any],
    run_name: str | None = None,
    timestamp: str | None = None,
) -> str:
    parts = [_format_run_timestamp(timestamp), build_lora_run_core_name(config)]
    if run_name is not None and str(run_name).strip():
        parts.append(_sanitize_name_part(run_name, fallback="run"))
    return "_".join(parts)


def _format_run_timestamp(timestamp: str | None = None) -> str:
    if timestamp is None:
        return time.strftime("%m%d%H%M%S")

    digits = re.sub(r"\D", "", str(timestamp))
    if len(digits) == 10:
        return digits
    if len(digits) == 14:
        return digits[4:]
    raise ValueError("Log timestamp must be mmddhhmmss or yyyymmddhhmmss.")


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def _require_value(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] in (None, ""):
        raise ValueError(f"Config value 'LoRA.{key}' is required for log naming.")
    return config[key]


def _extract_model_size(model_name: Any) -> str:
    match = _MODEL_SIZE_PATTERN.search(str(model_name or ""))
    if not match:
        raise ValueError(
            "global.model_name must contain a model size like 4B, 2B, or 0.8B "
            "for LoRA log naming."
        )
    return match.group(1).replace("b", "B")


def _normalize_scaling_type(value: Any) -> str:
    return str(value).strip().lower().replace("_", "/") or "r/a"


def _build_layer_selection_name(lora_cfg: Mapping[str, Any]) -> str:
    side = str(lora_cfg.get("target_layer_side", "all")).strip().lower()
    if side in {"", "all", "none", "full"}:
        return "full"

    if side in {"front", "first", "input", "top"}:
        prefix = "top"
    elif side in {"back", "last", "output", "end"}:
        prefix = "end"
    else:
        raise ValueError("LoRA.target_layer_side must be one of: all, input, output.")

    count = int(_require_value(lora_cfg, "target_layer_count"))
    if count <= 0:
        raise ValueError(
            "LoRA.target_layer_count must be greater than 0 when "
            "LoRA.target_layer_side is input/output."
        )
    return f"{prefix}-{count}"


def _build_target_module_name(target_modules: Any) -> str:
    if isinstance(target_modules, str):
        modules = [
            module.strip()
            for module in re.split(r"[, ]+", target_modules)
            if module.strip()
        ]
    elif isinstance(target_modules, Sequence):
        modules = [str(module).strip() for module in target_modules if str(module).strip()]
    else:
        raise ValueError("LoRA.target_modules must be a non-empty list or string.")

    if not modules:
        raise ValueError("LoRA.target_modules must not be empty.")

    enabled = []
    for letter, aliases in _TARGET_MODULE_ALIASES.items():
        if any(_matches_target_alias(module, aliases) for module in modules):
            enabled.append(letter)

    if enabled:
        return "".join(enabled)

    compact_modules = [
        _sanitize_name_part(module.rsplit(".", 1)[-1], fallback="module")
        for module in modules
    ]
    return "-".join(dict.fromkeys(compact_modules))


def _matches_target_alias(module: str, aliases: tuple[str, ...]) -> bool:
    basename = module.rsplit(".", 1)[-1].strip().lower()
    return basename in aliases


def _to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric for log naming.") from exc


def _format_number(value: Any) -> str:
    number = _to_float(value, "number")
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _sanitize_name_part(value: str, fallback: str) -> str:
    safe_value = re.sub(r"[\\/:\s=]+", "-", str(value).strip())
    return safe_value.strip("-_") or fallback


class Logger:
    def __init__(
        self,
        log_path: str | None = None,
        run_name: str | None = None,
        finetuning_type: str | None = None,
        log_file: str | None = None,
        tensorboard_logdir: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.tensorboard_logdir = None
        self.tensorboard_writer = None
        if log_file is None:
            if log_path is not None and finetuning_type is not None:
                run_dir_name = self._build_run_dir_name(
                    finetuning_type,
                    run_name,
                    config,
                )
                self.run_dir_name = run_dir_name
                self.run_dir = os.path.join(log_path, run_dir_name)
                self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
                self.final_model_dir = os.path.join(self.checkpoint_dir, "model-final")
                log_file = os.path.join(self.run_dir, "log.txt")
            elif run_name is None and log_path and log_path.endswith(".txt"):
                log_file = log_path
            else:
                raise ValueError(
                    "Either log_file or both log_path and finetuning_type must be set."
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
        self.best_model_dir = os.path.join(self.checkpoint_dir, "model-best-testloss")
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if tensorboard_logdir is not None:
            self._init_tensorboard(tensorboard_logdir)

    def _build_run_dir_name(
        self,
        finetuning_type: str,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> str:
        if str(finetuning_type).strip().lower() == "lora" and config is not None:
            return build_lora_run_dir_name(config, run_name=run_name)

        timestamp = time.strftime("%m%d%H%M%S")
        parts = [
            timestamp,
            self._sanitize_run_dir_part(finetuning_type, fallback="finetuning"),
        ]
        if run_name is not None and str(run_name).strip():
            parts.append(self._sanitize_run_dir_part(run_name, fallback="run"))
        return "_".join(parts)

    def _sanitize_run_dir_part(self, value: str, fallback: str) -> str:
        safe_value = re.sub(r"[\\/:\s]+", "_", str(value).strip())
        return safe_value.strip("_") or fallback

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
            os.path.expanduser(tensorboard_logdir),
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
