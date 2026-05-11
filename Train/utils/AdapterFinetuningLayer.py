from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from peft import AdaptionPromptConfig, TaskType

from .FinetuneMethod import FinetuneMethod


class AdapterFinetuningMethod(FinetuneMethod):
    method_name = "AdapterFinetuning"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.target_modules = str(self.config["target_modules"])
        self.adapter_len = int(self.config["adapter_len"])
        self.adapter_layers = int(self.config["adapter_layers"])

    def build_config(self) -> AdaptionPromptConfig:
        return AdaptionPromptConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
            adapter_len=self.adapter_len,
            adapter_layers=self.adapter_layers,
        )

    def default_run_name(self) -> str:
        return (
            f"adapter_len={self.adapter_len}_"
            f"layers={self.adapter_layers}_target={self.target_modules}"
        )
