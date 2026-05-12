from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from peft import LoraConfig, TaskType

from .FinetuneMethod import FinetuneMethod


class LoRAFinetuneMethod(FinetuneMethod):
    method_name = "LoRA"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.rank = int(self.config["rank"])
        self.alpha = float(self.config["alpha"])
        self.scaling_type = str(self.config.get("scaling_type", "r/a"))
        self.dropout_rate = float(self.config.get("dropout_rate", 0.0))
        self.target_modules = list(self.config["target_modules"])
        self.init_lora_weights = self.config.get("init_lora_weights", "gaussian")

    def build_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout_rate,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=(self.scaling_type == "r/sqrta"),
        )

    def default_run_name(self) -> str:
        return (
            f"lora_r={self.rank}_alpha={self.alpha}_"
            f"scaling={self.scaling_type}_dropout={self.dropout_rate}"
        )
