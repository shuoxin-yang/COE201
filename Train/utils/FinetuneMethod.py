from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from peft import get_peft_model


class FinetuneMethod(ABC):
    method_name: str

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)

    @abstractmethod
    def build_config(self):
        raise NotImplementedError

    @abstractmethod
    def default_run_name(self) -> str:
        raise NotImplementedError

    def apply(self, model):
        return get_peft_model(model, self.build_config())
