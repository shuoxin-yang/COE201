from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"


def argparser():
    parser = argparse.ArgumentParser(description="Interactive QA assistant eval script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3.6-27B",
        help="Base model path or model directory name under Project/model",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional PEFT adapter path. Supports LoRA, PrefixFT, and AdapterFinetuning checkpoints.",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
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
        return str(model_path)

    project_model_path = MODEL_ROOT / model_name
    if project_model_path.is_dir():
        return str(project_model_path)

    return model_name


def resolve_adapter_path(adapter_path: str | None) -> str | None:
    if adapter_path is None:
        return None

    path = Path(adapter_path).expanduser()
    if not path.exists():
        project_path = PROJECT_ROOT / adapter_path
        if project_path.exists():
            path = project_path
    if not path.exists():
        raise FileNotFoundError(f"PEFT adapter path does not exist: {adapter_path}")
    if path.is_file():
        raise ValueError(f"PEFT adapter path must be a directory: {adapter_path}")
    if (path / "adapter_config.json").is_file():
        return str(path)

    adapter_configs = sorted(
        path.rglob("adapter_config.json"),
        key=lambda config_path: adapter_sort_key(config_path.parent),
    )
    if not adapter_configs:
        raise FileNotFoundError(
            f"No adapter_config.json found under PEFT adapter path: {adapter_path}"
        )
    return str(adapter_configs[-1].parent)


def adapter_info(adapter_path: str | None) -> dict:
    if adapter_path is None:
        return {}

    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return {
        "peft_type": config.get("peft_type", "unknown"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "path": adapter_path,
    }


def adapter_sort_key(path: Path) -> tuple[int, int, float, str]:
    numbers = re.findall(r"\d+", path.name)
    last_number = int(numbers[-1]) if numbers else -1
    return (int(bool(numbers)), last_number, path.stat().st_mtime, str(path))


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


def load_model(model_path: str, adapter_path: str | None, dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


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
    model_path = resolve_model_path(args.model_name)
    adapter_path = resolve_adapter_path(args.adapter_path)
    peft_adapter_info = adapter_info(adapter_path)
    dtype = select_dtype(args.dtype, args.device)

    print("=" * 60)
    print("Course QA Assistant Eval")
    print("=" * 60)
    print(f"Base model: {model_path}")
    print(f"PEFT adapter: {adapter_path if adapter_path else 'None'}")
    if peft_adapter_info:
        print(f"PEFT type: {peft_adapter_info['peft_type']}")
    print(f"Device: {args.device}")
    print(f"Dtype: {dtype}")

    tokenizer = setup_tokenizer(model_path)
    print(f"Tokenizer EOS ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})")

    print("\nLoading model...")
    model = load_model(model_path, adapter_path, dtype, args.device)
    if adapter_path is not None:
        print("PEFT adapter loaded.")
    print("Model loaded.")

    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
