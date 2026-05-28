from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model

from .FinetuneMethod import FinetuneMethod


class LoRAFinetuneMethod(FinetuneMethod):
    method_name = "LoRA"
    _LAYER_CONTAINER_NAMES = {
        "block",
        "blocks",
        "decoder_layers",
        "encoder_layers",
        "h",
        "layer",
        "layers",
    }

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.rank = int(self.config["rank"])
        self.alpha = float(self.config["alpha"])
        self.scaling_type = str(self.config.get("scaling_type", "r/a"))
        self.dropout_rate = float(self.config.get("dropout_rate", 0.0))
        self.target_modules = list(self.config["target_modules"])
        self.target_layer_side = self._normalize_target_layer_side(
            self.config.get("target_layer_side", "all")
        )
        self.target_layer_count = self._parse_target_layer_count(
            self.config.get("target_layer_count", 0)
        )
        if self.target_layer_side == "all" and self.target_layer_count != 0:
            raise ValueError(
                "LoRA.target_layer_count must be 0 when "
                "LoRA.target_layer_side is all."
            )
        self.init_lora_weights = self.config.get("init_lora_weights", "gaussian")
        self.resolved_target_modules = list(self.target_modules)
        self.resolved_target_layers: list[str] = []

    def build_config(self) -> LoraConfig:
        return self._build_config(self.target_modules)

    def apply(self, model):
        self.resolved_target_modules = self._resolve_target_modules(model)
        return get_peft_model(model, self._build_config(self.resolved_target_modules))

    def target_summary(self) -> dict[str, Any]:
        return {
            "target_layer_side": self.target_layer_side,
            "target_layer_count": self.target_layer_count,
            "target_layers": self.resolved_target_layers,
            "target_modules": self.resolved_target_modules,
        }

    def _build_config(self, target_modules: list[str]) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout_rate,
            init_lora_weights=self.init_lora_weights,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=(self.scaling_type == "r/sqrta"),
        )

    def _resolve_target_modules(self, model) -> list[str]:
        self.resolved_target_layers = []
        if self.target_layer_side == "all":
            return list(self.target_modules)

        if self.target_layer_count <= 0:
            raise ValueError(
                "LoRA.target_layer_count must be greater than 0 when "
                "LoRA.target_layer_side is 'input' or 'output'."
            )

        grouped_targets = self._find_target_modules_by_layer(model)
        if not grouped_targets:
            raise ValueError(
                "LoRA layer selection found no matching target modules for "
                f"{self.target_modules}."
            )
        if self.target_layer_count > len(grouped_targets):
            raise ValueError(
                "LoRA.target_layer_count is greater than matched attention blocks: "
                f"requested {self.target_layer_count}, found {len(grouped_targets)}."
            )

        if self.target_layer_side == "input":
            selected_groups = grouped_targets[: self.target_layer_count]
        else:
            selected_groups = grouped_targets[-self.target_layer_count :]

        self.resolved_target_layers = [layer for layer, _ in selected_groups]
        return [
            module_name
            for _, module_names in selected_groups
            for module_name in module_names
        ]

    def _find_target_modules_by_layer(self, model) -> list[tuple[str, list[str]]]:
        groups: dict[str, list[str]] = {}
        for module_name, _ in model.named_modules():
            if not module_name or not self._matches_target_module(module_name):
                continue

            layer_name = self._infer_layer_name(module_name)
            groups.setdefault(layer_name, []).append(module_name)
        return list(groups.items())

    def _matches_target_module(self, module_name: str) -> bool:
        return any(
            module_name == target_module or module_name.endswith(f".{target_module}")
            for target_module in self.target_modules
        )

    @classmethod
    def _infer_layer_name(cls, module_name: str) -> str:
        parts = module_name.split(".")
        for idx, part in enumerate(parts):
            if idx == 0 or not part.isdigit():
                continue

            previous = parts[idx - 1]
            if (
                previous in cls._LAYER_CONTAINER_NAMES
                or previous.endswith("layers")
                or previous.endswith("blocks")
            ):
                return ".".join(parts[: idx + 1])

        for idx, part in enumerate(parts):
            if part.isdigit():
                return ".".join(parts[: idx + 1])

        return module_name.rsplit(".", 1)[0] if "." in module_name else module_name

    @staticmethod
    def _normalize_target_layer_side(value: Any) -> str:
        side = str(value).strip().lower()
        if side in {"", "all", "none"}:
            return "all"
        if side in {"front", "first", "input"}:
            return "input"
        if side in {"back", "last", "output"}:
            return "output"
        raise ValueError(
            "LoRA.target_layer_side must be one of: all, input, output."
        )

    @staticmethod
    def _parse_target_layer_count(value: Any) -> int:
        if value is None or value == "":
            return 0
        return int(value)

    def default_run_name(self) -> str:
        run_name = (
            f"lora_r={self.rank}_alpha={self.alpha}_"
            f"scaling={self.scaling_type}_dropout={self.dropout_rate}"
        )
        if self.target_layer_side != "all":
            run_name += (
                f"_layers={self.target_layer_side}{self.target_layer_count}"
            )
        return run_name
