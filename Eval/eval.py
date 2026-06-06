from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
TRAIN_DIR = PROJECT_ROOT / "Train"
DEFAULT_MODEL_NAME = "Qwen3.5-2B"
PEFT_ADAPTER_CONFIG = "adapter_config.json"
TRAINING_CONFIG = "config.yaml"
SAFE_WEIGHTS = "model.safetensors"
PYTORCH_WEIGHTS = "pytorch_model.bin"
SAFE_WEIGHTS_INDEX = "model.safetensors.index.json"
PYTORCH_WEIGHTS_INDEX = "pytorch_model.bin.index.json"
COMMON_STATE_DICT_PREFIXES = (
    "module.",
    "_orig_mod.",
    "base_model.model.",
    "model.",
)
COMMON_STATE_DICT_REPLACEMENTS = (
    ("model.language_model.", "model."),
    ("language_model.", "model."),
)


@dataclass(frozen=True)
class AdapterLoadSpec:
    kind: str
    path: Path
    method_name: str | None = None
    method_config: dict[str, Any] | None = None
    base_model_name_or_path: str | None = None


def argparser():
    parser = argparse.ArgumentParser(description="Interactive QA assistant eval script")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Base model path or model directory name under Project/model. "
            "When a training checkpoint records its base model and this argument "
            f"is left as the default ({DEFAULT_MODEL_NAME}), that recorded model "
            "is used automatically."
        ),
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help=(
            "Optional adapter/checkpoint path. Supports PEFT LoRA/PrefixFT adapter "
            "directories and custom AdapterFinetuning Trainer checkpoints."
        ),
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是离散数学课程助教。请用准确、清晰、适合学生理解的方式回答课程相关问题。",
        help="Optional system prompt used before each conversation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p value.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for repeated text.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Model dtype used for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--no_history",
        action="store_true",
        help="Do not keep previous turns in the prompt.",
    )
    return parser.parse_args()


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        return str(model_path.resolve())

    if not model_path.is_absolute():
        project_relative_path = PROJECT_ROOT / model_path
        if project_relative_path.is_dir():
            return str(project_relative_path.resolve())

    project_model_path = MODEL_ROOT / model_name
    if project_model_path.is_dir():
        return str(project_model_path.resolve())

    return model_name


def resolve_existing_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            PROJECT_ROOT / path,
            TRAIN_DIR / path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Adapter/checkpoint path does not exist: {path_value}")


def resolve_adapter_checkpoint(adapter_path: str | None) -> AdapterLoadSpec | None:
    if adapter_path is None:
        return None

    path = resolve_existing_path(adapter_path)
    if path.is_file():
        raise ValueError(f"Adapter/checkpoint path must be a directory: {adapter_path}")

    peft_path = find_peft_adapter_path(path)
    if peft_path is not None:
        return AdapterLoadSpec(
            kind="peft",
            path=peft_path,
            method_name=read_peft_type(peft_path),
            base_model_name_or_path=read_peft_base_model(peft_path),
        )

    adapter_finetuning_path = find_adapter_finetuning_checkpoint(path)
    if adapter_finetuning_path is not None:
        method_config, base_model_name = read_adapter_finetuning_config(
            adapter_finetuning_path
        )
        return AdapterLoadSpec(
            kind="adapter_finetuning",
            path=adapter_finetuning_path,
            method_name="AdapterFinetuning",
            method_config=method_config,
            base_model_name_or_path=base_model_name,
        )

    raise FileNotFoundError(
        "No PEFT adapter_config.json or AdapterFinetuning model weights found "
        f"under adapter/checkpoint path: {adapter_path}"
    )


def find_peft_adapter_path(path: Path) -> Path | None:
    if (path / PEFT_ADAPTER_CONFIG).is_file():
        return path
    adapter_configs = sorted(
        path.rglob(PEFT_ADAPTER_CONFIG),
        key=lambda config_path: adapter_sort_key(config_path.parent),
    )
    if not adapter_configs:
        return None
    return adapter_configs[-1].parent


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_peft_type(path: Path) -> str:
    config = read_json(path / PEFT_ADAPTER_CONFIG)
    return str(config.get("peft_type", "unknown"))


def read_peft_base_model(path: Path) -> str | None:
    config = read_json(path / PEFT_ADAPTER_CONFIG)
    base_model = config.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def has_model_weights(path: Path) -> bool:
    return any(
        (path / filename).is_file()
        for filename in (
            SAFE_WEIGHTS,
            PYTORCH_WEIGHTS,
            SAFE_WEIGHTS_INDEX,
            PYTORCH_WEIGHTS_INDEX,
        )
    )


def find_adapter_finetuning_checkpoint(path: Path) -> Path | None:
    if has_model_weights(path) and is_adapter_finetuning_checkpoint(path):
        return path

    candidates = []
    seen = set()
    for config_path in path.rglob("config.json"):
        candidate = config_path.parent
        if candidate in seen or not has_model_weights(candidate):
            continue
        seen.add(candidate)
        if is_adapter_finetuning_checkpoint(candidate):
            candidates.append(candidate)

    if not candidates:
        return None
    return sorted(candidates, key=adapter_sort_key)[-1]


def is_adapter_finetuning_checkpoint(path: Path) -> bool:
    method = infer_training_method(path)
    if method is not None:
        return method == "AdapterFinetuning"
    return state_dict_has_adapter_finetuning_keys(path)


def infer_training_method(path: Path) -> str | None:
    training_config_path = find_training_config_path(path)
    run_dir = training_config_path.parent if training_config_path is not None else None
    if run_dir is not None:
        log_path = run_dir / "log.txt"
        if log_path.is_file():
            match = re.search(
                r"^\s*active_method:\s*(\S+)\s*$",
                log_path.read_text(encoding="utf-8", errors="ignore"),
                flags=re.MULTILINE,
            )
            if match:
                return normalize_method_name(match.group(1))

    for part in reversed(path.parts):
        normalized = normalize_method_name(part)
        if normalized is not None:
            return normalized
    return None


def normalize_method_name(value: str) -> str | None:
    method_name = str(value).lower()
    if "adapterfinetuning" in method_name or "adapter_finetuning" in method_name:
        return "AdapterFinetuning"
    if "prefixft" in method_name or "prefix" in method_name:
        return "PrefixFT"
    if "lora" in method_name:
        return "LoRA"
    return None


def find_training_config_path(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        candidate = directory / TRAINING_CONFIG
        if candidate.is_file():
            return candidate
    return None


def read_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config loading requires pyyaml to be installed.") from exc

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, Mapping):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return dict(config)


def read_adapter_finetuning_config(path: Path) -> tuple[dict[str, Any], str | None]:
    training_config_path = find_training_config_path(path)
    if training_config_path is None:
        raise FileNotFoundError(
            "AdapterFinetuning checkpoints require the training run config.yaml "
            f"next to the log directory. Could not find it for: {path}"
        )

    training_config = read_yaml_config(training_config_path)
    method_config = training_config.get("AdapterFinetuning")
    if not isinstance(method_config, Mapping):
        raise ValueError(
            "Training config does not contain an AdapterFinetuning section: "
            f"{training_config_path}"
        )

    global_config = training_config.get("global", {})
    base_model = None
    if isinstance(global_config, Mapping) and global_config.get("model_name"):
        base_model = str(global_config["model_name"])
    return dict(method_config), base_model


def state_dict_has_adapter_finetuning_keys(path: Path) -> bool:
    key_sources = []
    for index_name in (SAFE_WEIGHTS_INDEX, PYTORCH_WEIGHTS_INDEX):
        index_path = path / index_name
        if index_path.is_file():
            try:
                weight_map = read_json(index_path).get("weight_map", {})
            except (OSError, json.JSONDecodeError):
                weight_map = {}
            key_sources.append(weight_map.keys())

    direct_safe_weights = path / SAFE_WEIGHTS
    if direct_safe_weights.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(direct_safe_weights), framework="pt", device="cpu") as f:
                key_sources.append(f.keys())
        except ImportError:
            pass

    return any(
        ".adapter." in key or key.endswith(".adapter.gate")
        for keys in key_sources
        for key in keys
    )


def adapter_info(adapter_spec: AdapterLoadSpec | None) -> dict:
    if adapter_spec is None:
        return {}

    return {
        "type": adapter_spec.method_name or adapter_spec.kind,
        "base_model_name_or_path": adapter_spec.base_model_name_or_path,
        "path": str(adapter_spec.path),
    }


def adapter_sort_key(path: Path) -> tuple[int, int, float, str]:
    numbers = re.findall(r"\d+", path.name)
    last_number = int(numbers[-1]) if numbers else -1
    return (int(bool(numbers)), last_number, path.stat().st_mtime, str(path))


def selected_model_name(model_name: str, adapter_spec: AdapterLoadSpec | None) -> str:
    if (
        adapter_spec is not None
        and adapter_spec.base_model_name_or_path
        and model_name == DEFAULT_MODEL_NAME
    ):
        return adapter_spec.base_model_name_or_path
    return model_name


def select_dtype(dtype: str, device: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32" or device == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def setup_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: str,
    adapter_spec: AdapterLoadSpec | None,
    dtype,
    device: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_spec is not None:
        if adapter_spec.kind == "peft":
            model = PeftModel.from_pretrained(model, str(adapter_spec.path))
        elif adapter_spec.kind == "adapter_finetuning":
            model = load_adapter_finetuning_checkpoint(model, adapter_spec)
        else:
            raise ValueError(f"Unsupported adapter checkpoint type: {adapter_spec.kind}")

    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


def load_adapter_finetuning_checkpoint(model, adapter_spec: AdapterLoadSpec):
    if adapter_spec.method_config is None:
        raise ValueError("AdapterFinetuning checkpoint is missing method config.")

    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    from utils.AdapterFinetuningLayer import AdapterFinetuningMethod

    model = AdapterFinetuningMethod(adapter_spec.method_config).apply(model)
    load_checkpoint_weights(model, adapter_spec.path)
    return model


def load_checkpoint_weights(model, checkpoint_dir: Path) -> None:
    model_keys = set(model.state_dict().keys())
    index_path = first_existing(
        checkpoint_dir,
        (SAFE_WEIGHTS_INDEX, PYTORCH_WEIGHTS_INDEX),
    )
    if index_path is not None:
        load_sharded_state_dict(model, checkpoint_dir, index_path, model_keys)
        return

    weights_path = first_existing(checkpoint_dir, (SAFE_WEIGHTS, PYTORCH_WEIGHTS))
    if weights_path is None:
        raise FileNotFoundError(f"No model weights found in checkpoint: {checkpoint_dir}")

    state_dict = load_weight_file(weights_path)
    key_transform, transform_name = select_key_transform(
        model_keys=model_keys,
        checkpoint_keys=state_dict.keys(),
    )
    loaded_keys, unexpected_keys = load_transformed_state_dict(
        model=model,
        state_dict=state_dict,
        model_keys=model_keys,
        key_transform=key_transform,
    )
    report_checkpoint_load(
        model_keys=model_keys,
        loaded_keys=loaded_keys,
        unexpected_keys=unexpected_keys,
        transform_name=transform_name,
    )


def first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def load_sharded_state_dict(
    model,
    checkpoint_dir: Path,
    index_path: Path,
    model_keys: set[str],
) -> None:
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"Invalid sharded checkpoint index: {index_path}")

    key_transform, transform_name = select_key_transform(
        model_keys=model_keys,
        checkpoint_keys=weight_map.keys(),
    )
    loaded_keys = set()
    unexpected_keys = []
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        state_dict = load_weight_file(shard_path)
        shard_loaded_keys, shard_unexpected_keys = load_transformed_state_dict(
            model=model,
            state_dict=state_dict,
            model_keys=model_keys,
            key_transform=key_transform,
        )
        loaded_keys.update(shard_loaded_keys)
        unexpected_keys.extend(shard_unexpected_keys)

    report_checkpoint_load(
        model_keys=model_keys,
        loaded_keys=loaded_keys,
        unexpected_keys=unexpected_keys,
        transform_name=transform_name,
    )


def load_weight_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Loading safetensors checkpoints requires safetensors to be installed."
            ) from exc
        return load_file(str(path), device="cpu")

    state_dict = torch.load(str(path), map_location="cpu")
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint file did not contain a state dict: {path}")
    return dict(state_dict)


def select_key_transform(
    model_keys: set[str],
    checkpoint_keys,
):
    candidates = build_key_transform_candidates(model_keys)
    scores = []
    for name, transform in candidates:
        matched = sum(1 for key in checkpoint_keys if transform(key) in model_keys)
        scores.append((matched, name, transform))

    matched, transform_name, key_transform = max(scores, key=lambda item: item[0])
    if transform_name != "identity":
        print(
            "Checkpoint key transform: "
            f"{transform_name} ({matched} matched tensors)."
        )
    return key_transform, transform_name


def build_key_transform_candidates(model_keys: set[str]):
    prefix_ops = [("identity", lambda key: key)]
    for prefix in COMMON_STATE_DICT_PREFIXES:
        prefix_ops.append(
            (
                f"strip '{prefix}'",
                lambda key, prefix=prefix: key[len(prefix) :]
                if key.startswith(prefix)
                else key,
            )
        )
        prefix_ops.append((f"add '{prefix}'", lambda key, prefix=prefix: prefix + key))
    for old_prefix, new_prefix in COMMON_STATE_DICT_REPLACEMENTS:
        prefix_ops.append(
            (
                f"replace '{old_prefix}' with '{new_prefix}'",
                lambda key, old_prefix=old_prefix, new_prefix=new_prefix: (
                    new_prefix + key[len(old_prefix) :]
                    if key.startswith(old_prefix)
                    else key
                ),
            )
        )

    wrapper_ops = [
        ("identity", lambda key: key),
        ("insert adapter module wrapper", insert_adapter_module_key),
        ("remove adapter module wrapper", remove_adapter_module_key),
        (
            "prefer existing adapter module wrapper",
            lambda key: prefer_existing_key(
                key,
                (insert_adapter_module_key(key), remove_adapter_module_key(key)),
                model_keys,
            ),
        ),
        (
            "unique suffix match",
            lambda key: find_unique_suffix_match(key, model_keys),
        ),
    ]

    candidates = []
    for prefix_name, prefix_op in prefix_ops:
        for wrapper_name, wrapper_op in wrapper_ops:
            if prefix_name == "identity" and wrapper_name == "identity":
                name = "identity"
            elif prefix_name == "identity":
                name = wrapper_name
            elif wrapper_name == "identity":
                name = prefix_name
            else:
                name = f"{prefix_name} + {wrapper_name}"
            candidates.append(
                (
                    name,
                    lambda key, prefix_op=prefix_op, wrapper_op=wrapper_op: wrapper_op(
                        prefix_op(key)
                    ),
                )
            )
    return candidates


def insert_adapter_module_key(key: str) -> str:
    marker = ".self_attn."
    wrapped_marker = ".self_attn.module."
    adapter_marker = ".self_attn.adapter."
    if (
        marker in key
        and wrapped_marker not in key
        and adapter_marker not in key
    ):
        return key.replace(marker, wrapped_marker, 1)
    return key


def remove_adapter_module_key(key: str) -> str:
    return key.replace(".self_attn.module.", ".self_attn.", 1)


def prefer_existing_key(
    key: str,
    candidates: tuple[str, ...],
    model_keys: set[str],
) -> str:
    if key in model_keys:
        return key
    for candidate in candidates:
        if candidate in model_keys:
            return candidate
    return key


def find_unique_suffix_match(key: str, model_keys: set[str]) -> str:
    key_variants = (
        key,
        insert_adapter_module_key(key),
        remove_adapter_module_key(key),
    )
    for key_variant in key_variants:
        if key_variant in model_keys:
            return key_variant

        parts = key_variant.split(".")
        for start in range(1, len(parts)):
            suffix = ".".join(parts[start:])
            matches = [model_key for model_key in model_keys if model_key.endswith(suffix)]
            if len(matches) == 1:
                return matches[0]
    return key


def load_transformed_state_dict(
    model,
    state_dict: Mapping[str, torch.Tensor],
    model_keys: set[str],
    key_transform,
) -> tuple[set[str], list[tuple[str, str]]]:
    transformed_state_dict = {}
    unexpected_keys = []
    for key, value in state_dict.items():
        transformed_key = key_transform(key)
        if transformed_key in model_keys:
            transformed_state_dict[transformed_key] = value
        else:
            unexpected_keys.append((key, transformed_key))

    if transformed_state_dict:
        model.load_state_dict(transformed_state_dict, strict=False)
    return set(transformed_state_dict.keys()), unexpected_keys


def report_checkpoint_load(
    model_keys: set[str],
    loaded_keys: set[str],
    unexpected_keys: list[tuple[str, str]],
    transform_name: str,
) -> None:
    missing_keys = sorted(model_keys - loaded_keys)
    print(
        "Checkpoint tensors loaded: "
        f"{len(loaded_keys)}/{len(model_keys)} model tensors matched."
    )

    if not loaded_keys:
        unexpected_examples = format_key_examples(
            [original for original, _ in unexpected_keys]
        )
        missing_examples = format_key_examples(missing_keys)
        raise RuntimeError(
            "Checkpoint weights did not match the model at all. "
            f"Selected key transform: {transform_name}. "
            f"Unexpected checkpoint key examples: {unexpected_examples}. "
            f"Model key examples: {missing_examples}"
        )

    if missing_keys:
        print(
            "Warning: "
            f"{len(missing_keys)} model tensors were not loaded from checkpoint."
        )
        print(f"Missing key examples: {format_key_examples(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} checkpoint keys were unexpected.")
        print(
            "Unexpected key examples: "
            f"{format_key_examples([original for original, _ in unexpected_keys])}"
        )


def format_key_examples(keys, limit: int = 5) -> str:
    examples = [str(key) for key in list(keys)[:limit]]
    return ", ".join(examples) if examples else "None"


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def stop_token_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.append(im_end_id)

    return sorted(set(ids))


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": stop_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def chat_loop(model, tokenizer, args) -> None:
    print("\nAssistant ready. Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 60)

    history = []
    while True:
        query = input("\nUser> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if not query:
            continue

        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        if not args.no_history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        print("\nAssistant> ", end="", flush=True)
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(response)
        print("-" * 60)

        if not args.no_history:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})


def main():
    args = argparser()
    adapter_spec = resolve_adapter_checkpoint(args.adapter_path)
    model_name = selected_model_name(args.model_name, adapter_spec)
    model_path = resolve_model_path(model_name)
    adapter_metadata = adapter_info(adapter_spec)
    dtype = select_dtype(args.dtype, args.device)

    print("=" * 60)
    print("Course QA Assistant Eval")
    print("=" * 60)
    print(f"Base model: {model_path}")
    print(
        "Adapter/checkpoint: "
        f"{adapter_metadata['path'] if adapter_metadata else 'None'}"
    )
    if adapter_metadata:
        print(f"Adapter type: {adapter_metadata['type']}")
        if adapter_metadata.get("base_model_name_or_path"):
            print(f"Adapter base model: {adapter_metadata['base_model_name_or_path']}")
    print(f"Device: {args.device}")
    print(f"Dtype: {dtype}")

    tokenizer = setup_tokenizer(model_path)
    print(f"Tokenizer EOS ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})")

    print("\nLoading model...")
    model = load_model(model_path, adapter_spec, dtype, args.device)
    if adapter_spec is not None:
        print("Adapter/checkpoint loaded.")
    print("Model loaded.")

    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
