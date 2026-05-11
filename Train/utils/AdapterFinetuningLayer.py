from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .FinetuneMethod import FinetuneMethod


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_len: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_len, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(adapter_len, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_output = self.up(self.activation(self.down(hidden_states)))
        return hidden_states + self.gate * adapter_output


class AdapterWrappedModule(nn.Module):
    def __init__(self, module: nn.Module, hidden_size: int, adapter_len: int):
        super().__init__()
        self.module = module
        self.adapter = BottleneckAdapter(hidden_size, adapter_len)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, tuple):
            return (self.adapter(output[0]), *output[1:])
        return self.adapter(output)


class AdapterFinetuningMethod(FinetuneMethod):
    method_name = "AdapterFinetuning"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.target_modules = str(self.config["target_modules"])
        self.adapter_len = int(self.config["adapter_len"])
        self.adapter_layers = int(self.config["adapter_layers"])

    def build_config(self):
        return None

    def apply(self, model):
        for param in model.parameters():
            param.requires_grad = False

        targets = self._find_target_modules(model)
        if len(targets) < self.adapter_layers:
            raise ValueError(
                "AdapterFinetuning.adapter_layers is greater than matched target "
                f"modules: requested {self.adapter_layers}, found {len(targets)} "
                f"matching '{self.target_modules}'."
            )

        for name, module in targets[-self.adapter_layers :]:
            parent, child_name = self._resolve_parent_module(model, name)
            hidden_size = self._resolve_hidden_size(module, model)
            setattr(
                parent,
                child_name,
                AdapterWrappedModule(module, hidden_size, self.adapter_len),
            )
        return model

    def _find_target_modules(self, model) -> list[tuple[str, nn.Module]]:
        return [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(self.target_modules)
            and not isinstance(module, AdapterWrappedModule)
        ]

    def _resolve_parent_module(self, model, module_name: str) -> tuple[nn.Module, str]:
        parent = model
        path = module_name.split(".")
        for name in path[:-1]:
            parent = getattr(parent, name)
        return parent, path[-1]

    def _resolve_hidden_size(self, module: nn.Module, model) -> int:
        for owner in (module, getattr(model, "config", None)):
            hidden_size = getattr(owner, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)

        text_config = getattr(getattr(model, "config", None), "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

        raise ValueError(
            "Cannot infer hidden_size for AdapterFinetuning. "
            "Set target_modules to modules that expose hidden_size or use a model "
            "config with hidden_size/text_config.hidden_size."
        )

    def default_run_name(self) -> str:
        return (
            f"adapter_len={self.adapter_len}_"
            f"layers={self.adapter_layers}_target={self.target_modules}"
        )
