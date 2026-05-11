from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from peft import PrefixTuningConfig, TaskType

from .FinetuneMethod import FinetuneMethod


class PrefixFTFinetuneMethod(FinetuneMethod):
    method_name = "PrefixFT"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.num_virtual_tokens = int(self.config["num_virtual_tokens"])
        self.prefix_projection = bool(self.config.get("prefix_projection", False))
        self.init_weights = self.config.get("init_weights", True)

    def build_config(self) -> PrefixTuningConfig:
        kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "num_virtual_tokens": self.num_virtual_tokens,
            "prefix_projection": self.prefix_projection,
        }
        if "init_weights" in inspect.signature(PrefixTuningConfig).parameters:
            kwargs["init_weights"] = self.init_weights
        return PrefixTuningConfig(**kwargs)

    def default_run_name(self) -> str:
        projection = str(self.prefix_projection).lower()
        return (
            f"prefix_vtokens={self.num_virtual_tokens}_"
            f"projection={projection}"
        )
