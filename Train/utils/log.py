from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any


class Logger:
    def __init__(
        self,
        log_path: str | None = None,
        run_name: str | None = None,
        log_file: str | None = None,
    ):
        if log_file is None:
            if run_name is None and log_path and log_path.endswith(".txt"):
                log_file = log_path
            elif log_path is not None and run_name is not None:
                log_file = f"{log_path}/{int(time.time())}_{run_name}.txt"
            else:
                raise ValueError(
                    "Either log_file or both log_path and run_name must be set."
                )

        self.log_file = log_file
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def onlylog(self, message: Any, name: str | None = None) -> None:
        with open(self.log_file, "a") as f:
            f.write(self.format_message(message, name) + "\n")

    def logandprint(self, message: Any, name: str | None = None) -> None:
        formatted = self.format_message(message, name)
        print(formatted)
        self.onlylog(formatted)

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
