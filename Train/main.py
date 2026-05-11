from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from utils.AdapterFinetuningLayer import AdapterFinetuningMethod
from utils.LoRALayer import LoRAFinetuneMethod
from utils.PrefixFTLayer import PrefixFTFinetuneMethod
from utils.data import load_qa_dataset
from utils.log import Logger


TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
DEFAULT_CONFIG_PATH = TRAIN_DIR / "config.yaml"
METHOD_TRAINING_KEYS = (
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
)


def format_metrics(metrics: dict) -> dict:
    formatted = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 6)
        formatted[key] = value
    return formatted


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_params


def argparser():
    parser = argparse.ArgumentParser(description="PEFT fine-tuning entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML training config",
    )
    parser.add_argument(
        "--method",
        type=normalize_method,
        choices=("LoRA", "PrefixFT", "AdapterFinetuning"),
        required=True,
        help="Fine-tuning method: LoRA, PrefixFT, or AdapterFinetuning",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name appended to the log directory",
    )
    args, override_args = parser.parse_known_args()
    try:
        args.config_overrides = parse_config_overrides(override_args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_override_value(raw_value: str) -> Any:
    if raw_value == "":
        return ""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("CLI config overrides require pyyaml to be installed.") from exc

    return yaml.safe_load(raw_value)


def parse_config_overrides(
    override_args: list[str],
) -> list[tuple[tuple[str, ...], Any]]:
    overrides = []
    for raw_arg in override_args:
        if not raw_arg.startswith("--") or "=" not in raw_arg:
            raise ValueError(
                f"Unknown argument '{raw_arg}'. Config overrides must use "
                "--section.key=value syntax, for example --LoRA.rank=16."
            )

        raw_path, raw_value = raw_arg[2:].split("=", 1)
        key_path = tuple(raw_path.split("."))
        if len(key_path) < 2 or any(not key for key in key_path):
            raise ValueError(
                f"Invalid config override '{raw_arg}'. Expected "
                "--section.key=value syntax, for example --global.seed=43."
            )

        overrides.append((key_path, parse_override_value(raw_value)))

    return overrides


def load_config(config_path: str) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config loading requires pyyaml to be installed.") from exc

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return dict(config), path


def apply_config_overrides(
    config: MutableMapping[str, Any],
    overrides: list[tuple[tuple[str, ...], Any]],
) -> list[dict[str, Any]]:
    applied = []
    for key_path, value in overrides:
        path_text = ".".join(key_path)
        cursor: MutableMapping[str, Any] = config

        for depth, key in enumerate(key_path[:-1], start=1):
            if key not in cursor:
                missing_path = ".".join(key_path[:depth])
                raise ValueError(
                    f"Unknown config override '{path_text}': '{missing_path}' "
                    "does not exist."
                )

            next_value = cursor[key]
            if not isinstance(next_value, MutableMapping):
                parent_path = ".".join(key_path[:depth])
                raise ValueError(
                    f"Invalid config override '{path_text}': '{parent_path}' "
                    "is not a mapping."
                )
            cursor = next_value

        final_key = key_path[-1]
        if final_key not in cursor:
            raise ValueError(
                f"Unknown config override '{path_text}': key '{final_key}' "
                "does not exist."
            )

        old_value = cursor[final_key]
        cursor[final_key] = value
        applied.append({"key": path_text, "old": old_value, "new": value})

    return applied


def require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return dict(value)


def require_keys(config: Mapping[str, Any], keys: tuple[str, ...], section: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Config section '{section}' is missing keys: {missing}")


def normalize_method(method: Any) -> str:
    method_name = str(method).strip().lower()
    if method_name == "lora":
        return "LoRA"
    if method_name in {"prefixft", "prefix", "prefix_tuning", "prefixtuning"}:
        return "PrefixFT"
    if method_name in {
        "adapter",
        "adapterft",
        "adapter_finetuning",
        "adapterfinetuning",
        "adaption_prompt",
        "adaptionprompt",
        "llama_adapter",
    }:
        return "AdapterFinetuning"
    raise ValueError(
        "--method must be one of 'LoRA', 'PrefixFT', or 'AdapterFinetuning'."
    )


def validate_training_config(config: Mapping[str, Any], section: str) -> None:
    require_keys(config, METHOD_TRAINING_KEYS, section)
    if int(config["num_epochs"]) < 1:
        raise ValueError(f"{section}.num_epochs must be at least 1.")
    if int(config["batch_size"]) < 1:
        raise ValueError(f"{section}.batch_size must be at least 1.")
    if int(config["gradient_accumulation_steps"]) < 1:
        raise ValueError(f"{section}.gradient_accumulation_steps must be at least 1.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError(f"{section}.learning_rate must be greater than 0.")


def validate_config(config: Mapping[str, Any], method: str) -> None:
    global_cfg = require_mapping(config, "global")
    lora_cfg = require_mapping(config, "LoRA")
    prefix_cfg = require_mapping(config, "PrefixFT")
    adapter_cfg = require_mapping(config, "AdapterFinetuning")

    require_keys(
        global_cfg,
        (
            "model_name",
            "train_data_path",
            "max_length",
            "train_size",
            "eval_size",
            "test_size",
            "seed",
            "output_dir",
            "tensorboard_logdir",
            "logging_steps",
            "test_eval_epochs",
            "save_steps",
            "save_total_limit",
        ),
        "global",
    )
    if int(global_cfg["test_eval_epochs"]) < 0:
        raise ValueError("global.test_eval_epochs must be greater than or equal to 0.")

    for name in ("train_size", "eval_size", "test_size"):
        value = float(global_cfg[name])
        if value < 0 or value > 1:
            raise ValueError(f"global.{name} must be in [0, 1].")
    if float(global_cfg["train_size"]) <= 0:
        raise ValueError("global.train_size must be greater than 0.")
    if float(global_cfg["eval_size"]) <= 0:
        raise ValueError("global.eval_size must be greater than 0 for epoch eval.")
    split_total = (
        float(global_cfg["train_size"])
        + float(global_cfg["eval_size"])
        + float(global_cfg["test_size"])
    )
    if not math.isclose(split_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "global.train_size, global.eval_size, and global.test_size must sum to 1.0."
        )

    require_keys(
        lora_cfg,
        ("rank", "alpha", "scaling_type", "target_modules", "init_lora_weights"),
        "LoRA",
    )
    if int(lora_cfg["rank"]) <= 0:
        raise ValueError("LoRA.rank must be greater than 0.")
    if float(lora_cfg["alpha"]) <= 0:
        raise ValueError("LoRA.alpha must be greater than 0.")
    target_modules = lora_cfg["target_modules"]
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("LoRA.target_modules must be a non-empty list.")
    if str(lora_cfg["scaling_type"]) not in {"r/a", "r/sqrta"}:
        raise ValueError("LoRA.scaling_type must be either 'r/a' or 'r/sqrta'.")
    validate_training_config(lora_cfg, "LoRA")

    require_keys(
        prefix_cfg,
        ("num_virtual_tokens", "prefix_projection", "init_weights"),
        "PrefixFT",
    )
    if int(prefix_cfg["num_virtual_tokens"]) <= 0:
        raise ValueError("PrefixFT.num_virtual_tokens must be greater than 0.")
    validate_training_config(prefix_cfg, "PrefixFT")

    require_keys(
        adapter_cfg,
        ("target_modules", "adapter_len", "adapter_layers"),
        "AdapterFinetuning",
    )
    if not str(adapter_cfg["target_modules"]).strip():
        raise ValueError("AdapterFinetuning.target_modules must be a non-empty string.")
    if int(adapter_cfg["adapter_len"]) <= 0:
        raise ValueError("AdapterFinetuning.adapter_len must be greater than 0.")
    if int(adapter_cfg["adapter_layers"]) <= 0:
        raise ValueError("AdapterFinetuning.adapter_layers must be greater than 0.")
    validate_training_config(adapter_cfg, "AdapterFinetuning")
    require_mapping(config, method)


def resolve_path(value: Any, config_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def build_global_args(
    config: Mapping[str, Any],
    config_path: Path,
    method: str,
) -> SimpleNamespace:
    global_cfg = require_mapping(config, "global")
    method_cfg = require_mapping(config, method)
    config_dir = config_path.parent
    resolved = dict(global_cfg)
    resolved["method"] = method
    for key in (
        "max_length",
        "seed",
        "logging_steps",
        "test_eval_epochs",
        "save_steps",
        "save_total_limit",
    ):
        resolved[key] = int(resolved[key])
    for key in ("num_epochs", "batch_size", "gradient_accumulation_steps"):
        resolved[key] = int(method_cfg[key])
    for key in ("train_size", "eval_size", "test_size"):
        resolved[key] = float(global_cfg[key])
    resolved["learning_rate"] = float(method_cfg["learning_rate"])
    resolved["train_data_path"] = resolve_path(global_cfg["train_data_path"], config_dir)
    resolved["output_dir"] = resolve_path(global_cfg["output_dir"], config_dir)
    resolved["tensorboard_logdir"] = resolve_path(
        global_cfg["tensorboard_logdir"],
        config_dir,
    )
    return SimpleNamespace(**resolved)


def build_finetune_method(config: Mapping[str, Any], method: str):
    if method == "LoRA":
        return LoRAFinetuneMethod(require_mapping(config, "LoRA"))
    if method == "PrefixFT":
        return PrefixFTFinetuneMethod(require_mapping(config, "PrefixFT"))
    if method == "AdapterFinetuning":
        return AdapterFinetuningMethod(require_mapping(config, "AdapterFinetuning"))
    raise ValueError(f"Unsupported finetune method: {method}")


def backup_config(config: Mapping[str, Any], run_dir: str) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config backup requires pyyaml to be installed.") from exc

    backup_path = Path(run_dir) / "config.yaml"
    with open(backup_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=True)
    return str(backup_path)


def build_training_arguments(
    args: SimpleNamespace,
    logger: Logger,
    has_eval_dataset: bool,
) -> TrainingArguments:
    training_args_kwargs = {
        "output_dir": logger.checkpoint_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": [],
    }

    if has_eval_dataset:
        signature = inspect.signature(TrainingArguments).parameters
        eval_strategy_arg = (
            "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
        )
        training_args_kwargs[eval_strategy_arg] = "epoch"

    return TrainingArguments(**training_args_kwargs)


def resolve_test_eval_epochs(interval: int, num_epochs: int) -> set[int]:
    if interval == 0:
        return {num_epochs}
    if interval < 0:
        raise ValueError("global.test_eval_epochs must be greater than or equal to 0.")

    epochs = set(range(interval, num_epochs + 1, interval))
    epochs.add(num_epochs)
    return epochs


def resolve_model_path(model_name: str, config_dir: Path) -> str:
    model_path = Path(model_name).expanduser()
    candidates = []
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.append(config_dir / model_path)
        candidates.append(PROJECT_ROOT / "model" / model_name)

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Model directory does not exist. Checked: {[str(path) for path in candidates]}"
    )


def iter_model_configs(model) -> list[Any]:
    configs = []
    seen = set()
    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(model, "model", None), "config", None),
    ]
    for config in candidates:
        if config is None or id(config) in seen:
            continue
        configs.append(config)
        seen.add(id(config))
        text_config = getattr(config, "text_config", None)
        if text_config is not None and id(text_config) not in seen:
            configs.append(text_config)
            seen.add(id(text_config))
    return configs


def disable_model_cache(model) -> None:
    for config in iter_model_configs(model):
        if hasattr(config, "use_cache"):
            config.use_cache = False


def model_has_linear_attention(model) -> bool:
    for config in iter_model_configs(model):
        layer_types = getattr(config, "layer_types", None)
        if layer_types and "linear_attention" in layer_types:
            return True
    return False


def validate_method_model_compatibility(finetune_method, model) -> None:
    if finetune_method.method_name == "PrefixFT" and model_has_linear_attention(model):
        raise RuntimeError(
            "PrefixFT is incompatible with this model's linear_attention layers. "
            "Prefix tuning passes past_key_values, but Qwen3.5 linear attention "
            "expects a linear-attention cache. Use --method LoRA or "
            "--method AdapterFinetuning for this model."
        )


def completed_epoch(state) -> int | None:
    if state.epoch is None:
        return None
    return int(round(state.epoch))


class MutableDatasetView(torch.utils.data.Dataset):
    def __init__(self, name: str):
        self.name = name
        self.dataset = None

    def set_dataset(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset())

    def __getitem__(self, index: int):
        return self._dataset()[index]

    def _dataset(self):
        if self.dataset is None:
            raise RuntimeError(f"{self.name} dataset has not been initialized.")
        return self.dataset


class TrainEvalResplitter:
    def __init__(
        self,
        dataset,
        train_size: float,
        eval_size: float,
        seed: int,
    ):
        train_eval_size = train_size + eval_size
        if train_size <= 0:
            raise ValueError("global.train_size must be greater than 0.")
        if eval_size <= 0:
            raise ValueError("global.eval_size must be greater than 0 for epoch eval.")
        self.dataset = dataset
        self.train_size_ratio = train_size
        self.eval_size_ratio = eval_size
        self.eval_split_size = eval_size / train_eval_size
        self.seed = seed
        self.split_id = -1
        self.train_dataset = MutableDatasetView("train")
        self.eval_dataset = MutableDatasetView("eval")
        self.resplit(split_id=0)

    def resplit(self, split_id: int | None = None) -> None:
        if split_id is None:
            split_id = self.split_id + 1
        split = self.dataset.train_test_split(
            test_size=self.eval_split_size,
            seed=self.seed + split_id,
            shuffle=True,
        )
        if len(split["train"]) == 0 or len(split["test"]) == 0:
            raise ValueError(
                "Train/eval split produced an empty split. "
                "Adjust train_size/eval_size or provide more data."
            )
        self.train_dataset.set_dataset(split["train"])
        self.eval_dataset.set_dataset(split["test"])
        self.split_id = split_id

    @property
    def train_size(self) -> int:
        return len(self.train_dataset)

    @property
    def eval_size_count(self) -> int:
        return len(self.eval_dataset)


class TrainerLoggerCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        metrics = {"global_step": state.global_step}
        metrics.update(format_metrics(logs))
        if any(key.startswith("test_") for key in logs):
            log_type = "Test"
        elif any(key.startswith("eval_") for key in logs):
            log_type = "Eval"
        else:
            log_type = "Train"
        self.logger.onlylog(
            metrics,
            name=f"Trainer {log_type} Log Step {state.global_step}",
        )
        self._log_tensorboard_curves(logs, state.global_step)

    def _log_tensorboard_curves(self, logs: dict, step: int) -> None:
        loss_metrics = {}
        if "loss" in logs:
            loss_metrics["train"] = logs["loss"]
        if "eval_loss" in logs:
            loss_metrics["eval"] = logs["eval_loss"]
        if "test_loss" in logs:
            loss_metrics["test"] = logs["test_loss"]
        if loss_metrics:
            self.logger.log_tensorboard(loss_metrics, step=step, prefix="loss")
        if "learning_rate" in logs:
            self.logger.log_tensorboard(
                {"learning_rate": logs["learning_rate"]},
                step=step,
            )


class BestTestLossCheckpointCallback(TrainerCallback):
    def __init__(self, logger: Logger, tokenizer):
        self.logger = logger
        self.tokenizer = tokenizer
        self.best_loss: float | None = None
        self.best_step: int | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "test_loss" not in metrics:
            return

        test_loss = self._to_float(metrics["test_loss"])
        if test_loss is None or not math.isfinite(test_loss):
            return
        if self.best_loss is not None and test_loss >= self.best_loss:
            return
        if not getattr(args, "should_save", True):
            return

        model = kwargs.get("model")
        if model is None:
            self.logger.onlylog(
                {"global_step": state.global_step, "test_loss": test_loss},
                name="Best Test Loss Checkpoint Skipped",
            )
            return

        self.best_loss = test_loss
        self.best_step = state.global_step
        self._save_best_checkpoint(model, state, metrics, test_loss)
        self.logger.logandprint(
            {
                "global_step": self.best_step,
                "test_loss": round(self.best_loss, 6),
                "checkpoint_dir": self.logger.best_model_dir,
            },
            name="Best Test Loss Checkpoint",
        )

    def _save_best_checkpoint(
        self,
        model,
        state,
        metrics: dict,
        test_loss: float,
    ) -> None:
        output_dir = self.logger.best_model_dir
        tmp_output_dir = f"{output_dir}.tmp"
        if os.path.isdir(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)
        os.makedirs(tmp_output_dir, exist_ok=True)

        model.save_pretrained(tmp_output_dir)
        self.tokenizer.save_pretrained(tmp_output_dir)
        self._save_metadata(tmp_output_dir, state, metrics, test_loss)

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.replace(tmp_output_dir, output_dir)

    def _save_metadata(
        self,
        output_dir: str,
        state,
        metrics: dict,
        test_loss: float,
    ) -> None:
        if hasattr(state, "save_to_json"):
            state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        metadata = {
            "global_step": state.global_step,
            "test_loss": test_loss,
            "metrics": format_metrics(metrics),
        }
        with open(
            os.path.join(output_dir, "best_checkpoint_metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _to_float(self, value) -> float | None:
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (TypeError, ValueError, RuntimeError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class FixedTestEvaluationCallback(TrainerCallback):
    def __init__(self, logger: Logger, test_dataset, test_eval_epochs: set[int]):
        self.logger = logger
        self.test_dataset = test_dataset
        self.test_eval_epochs = test_eval_epochs
        self.trainer: Trainer | None = None
        self.evaluated_epochs: set[int] = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return

        epoch = completed_epoch(state)
        if (
            epoch is None
            or epoch not in self.test_eval_epochs
            or epoch in self.evaluated_epochs
        ):
            return
        if self.trainer is None:
            raise RuntimeError("FixedTestEvaluationCallback.trainer is not attached.")

        self.evaluated_epochs.add(epoch)
        self.logger.logandprint(
            {"epoch": epoch, "test_samples": len(self.test_dataset)},
            name="Fixed Test Evaluation",
        )
        test_metrics = self.trainer.evaluate(
            eval_dataset=self.test_dataset,
            metric_key_prefix="test",
        )
        test_metrics["test_samples"] = len(self.test_dataset)
        test_metrics["epoch"] = epoch
        test_metrics["global_step"] = state.global_step
        self.trainer.save_metrics(f"test_epoch_{epoch}", test_metrics)
        self.logger.logandprint(
            format_metrics(test_metrics),
            name=f"Test Metrics Epoch {epoch}",
        )


class ResplitTrainEvalCallback(TrainerCallback):
    def __init__(self, logger: Logger, resplitter: TrainEvalResplitter):
        self.logger = logger
        self.resplitter = resplitter

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return

        epoch = completed_epoch(state)
        if epoch is None or epoch >= int(args.num_train_epochs):
            return

        self.resplitter.resplit(split_id=epoch)
        self.logger.logandprint(
            {
                "completed_epoch": epoch,
                "next_split_id": self.resplitter.split_id,
                "train_samples": self.resplitter.train_size,
                "eval_samples": self.resplitter.eval_size_count,
            },
            name="Train/Eval Resplit",
        )


def main():
    cli_args = argparser()
    method = cli_args.method
    config, config_path = load_config(cli_args.config)
    applied_overrides = apply_config_overrides(config, cli_args.config_overrides)
    validate_config(config, method)
    args = build_global_args(config, config_path, method)
    finetune_method = build_finetune_method(config, method)
    run_name = str(cli_args.run_name).strip() or None
    args.run_name = run_name
    test_eval_epochs = resolve_test_eval_epochs(
        int(args.test_eval_epochs),
        int(args.num_epochs),
    )
    logger = Logger(
        log_path=args.output_dir,
        run_name=run_name,
        finetuning_type=finetune_method.method_name,
        tensorboard_logdir=args.tensorboard_logdir,
    )
    config_backup_path = backup_config(config, logger.run_dir)
    assert (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ), "This script requires a GPU with bfloat16 support."
    device = torch.device("cuda")

    runtime_config = {
        "config_path": str(config_path),
        "config_backup_path": config_backup_path,
        "source_config": config,
        "cli_overrides": applied_overrides,
        "resolved_global": vars(args).copy(),
        "active_method": finetune_method.method_name,
        "run_dir": logger.run_dir,
        "log_file": logger.log_file,
        "checkpoint_dir": logger.checkpoint_dir,
        "final_model_dir": logger.final_model_dir,
        "best_model_dir": logger.best_model_dir,
        "resolved_test_eval_epochs": sorted(test_eval_epochs),
        "tensorboard_logdir": logger.tensorboard_logdir,
    }
    logger.onlylog(runtime_config, name="Config")

    model_path = resolve_model_path(str(args.model_name), config_path.parent)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.logandprint(
        f"Tokenizer native EOS ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})"
    )

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(
        device
    )
    disable_model_cache(model)
    validate_method_model_compatibility(finetune_method, model)
    model = finetune_method.apply(model).to(device)
    disable_model_cache(model)
    trainable_params, all_param = count_trainable_parameters(model)
    logger.logandprint(f"finetune method: {finetune_method.method_name}")
    logger.logandprint(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    dataset = load_qa_dataset(
        path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        train_size=float(args.train_size),
        eval_size=float(args.eval_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    resplitter = TrainEvalResplitter(
        dataset=dataset["train_eval"],
        train_size=float(args.train_size),
        eval_size=float(args.eval_size),
        seed=int(args.seed),
    )
    test_dataset = dataset["test"] if "test" in dataset else None
    logger.logandprint(f"train/eval pool samples: {len(dataset['train_eval'])}")
    logger.logandprint(f"initial train samples: {resplitter.train_size}")
    logger.logandprint(f"initial eval samples: {resplitter.eval_size_count}")
    if test_dataset is not None:
        logger.logandprint(f"fixed test samples: {len(test_dataset)}")
    training_args = build_training_arguments(
        args=args,
        logger=logger,
        has_eval_dataset=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    callbacks = [TrainerLoggerCallback(logger)]
    fixed_test_callback = None
    if test_dataset is not None:
        callbacks.append(BestTestLossCheckpointCallback(logger, tokenizer))
        if test_eval_epochs:
            fixed_test_callback = FixedTestEvaluationCallback(
                logger=logger,
                test_dataset=test_dataset,
                test_eval_epochs=test_eval_epochs,
            )
            callbacks.append(fixed_test_callback)
    callbacks.append(ResplitTrainEvalCallback(logger, resplitter))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=resplitter.train_dataset,
        eval_dataset=resplitter.eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    if fixed_test_callback is not None:
        fixed_test_callback.trainer = trainer
    disable_model_cache(model)

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    train_metrics["train_eval_pool_samples"] = len(dataset["train_eval"])
    train_metrics["final_train_samples"] = resplitter.train_size
    train_metrics["final_eval_samples"] = resplitter.eval_size_count
    train_metrics["global_step"] = train_result.global_step
    train_metrics["training_loss"] = train_result.training_loss
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    formatted_train_metrics = format_metrics(train_metrics)
    logger.logandprint(formatted_train_metrics, name="Train Metrics")

    trainer.save_model(logger.final_model_dir)

    logger.close()


if __name__ == "__main__":
    main()
