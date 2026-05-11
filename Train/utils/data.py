from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizerBase


DEFAULT_QA_SYSTEM_PROMPT = ""
DEFAULT_TYPE_FIELD = "type"


def _resolve_data_files(path: str) -> str | list[str]:
    data_path = Path(path)
    if data_path.is_file():
        return str(data_path)
    if data_path.is_dir():
        jsonl_files = sorted(
            str(file_path) for file_path in data_path.rglob("*.jsonl")
        )
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {path}")
        return jsonl_files
    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        user_message = messages[-1]["content"]
        return (
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _chat_end_token(tokenizer: PreTrainedTokenizerBase) -> str:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return "<|im_end|>"
    return tokenizer.eos_token or ""


def _build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    qa_type: str | None = None,
    system_prompt: str = DEFAULT_QA_SYSTEM_PROMPT,
) -> str:
    question = question.strip()
    if qa_type:
        question = f"问题类型：{qa_type.strip()}\n问题：{question}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    return _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )


def _build_response(tokenizer: PreTrainedTokenizerBase, answer: str) -> str:
    response = answer.strip() + _chat_end_token(tokenizer)
    if tokenizer.eos_token and not response.endswith(tokenizer.eos_token):
        response += tokenizer.eos_token
    return response


def _validate_qa_columns(
    column_names: list[str],
    question_field: str,
    answer_field: str,
) -> None:
    missing = [
        field
        for field in (question_field, answer_field)
        if field not in column_names
    ]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Available columns: {column_names}"
        )


def _validate_split_size(name: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"--{name} must be in [0, 1].")


def _validate_split_proportions(
    train_size: float,
    eval_size: float,
    test_size: float,
) -> None:
    for name, value in (
        ("train_size", train_size),
        ("eval_size", eval_size),
        ("test_size", test_size),
    ):
        _validate_split_size(name, value)
    if train_size <= 0:
        raise ValueError("--train_size must be greater than 0.")
    if eval_size <= 0:
        raise ValueError("--eval_size must be greater than 0 for epoch eval.")

    total = train_size + eval_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "--train_size, --eval_size, and --test_size must sum to 1.0."
        )


def _validate_split_lengths(
    train_eval_len: int,
    test_len: int | None,
    eval_size: float,
) -> None:
    if test_len is not None and test_len <= 0:
        raise ValueError("The fixed test split is empty. Decrease --test_size.")
    if train_eval_len <= 0:
        raise ValueError("The train/eval pool is empty. Decrease --test_size.")
    if eval_size > 0 and train_eval_len < 2:
        raise ValueError(
            "Need at least 2 samples in the train/eval pool when --eval_size is greater than 0."
        )


def load_qa_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    train_size: float = 0.7,
    eval_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    question_field: str = "question",
    answer_field: str = "answer",
    type_field: str | None = DEFAULT_TYPE_FIELD,
    include_type_in_prompt: bool = True,
    system_prompt: str = DEFAULT_QA_SYSTEM_PROMPT,
    data_format: str = "json",
) -> DatasetDict:
    """Load QA json/jsonl file(s) and return tokenized train/eval pool and fixed test split."""
    if path is None:
        raise ValueError("path must be provided.")
    if tokenizer is None:
        raise ValueError("tokenizer must be provided.")
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0.")

    data_files = _resolve_data_files(path)
    raw_dataset = hf_load_dataset(data_format, data_files=data_files)["train"]
    _validate_qa_columns(raw_dataset.column_names, question_field, answer_field)
    _validate_split_proportions(train_size, eval_size, test_size)
    use_type = (
        include_type_in_prompt
        and type_field is not None
        and type_field in raw_dataset.column_names
    )

    if test_size > 0:
        fixed_test_split = raw_dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            shuffle=True,
        )
        raw_splits = DatasetDict(
            {
                "train_eval": fixed_test_split["train"],
                "test": fixed_test_split["test"],
            }
        )
    else:
        raw_splits = DatasetDict({"train_eval": raw_dataset})
    _validate_split_lengths(
        train_eval_len=len(raw_splits["train_eval"]),
        test_len=len(raw_splits["test"]) if "test" in raw_splits else None,
        eval_size=eval_size,
    )

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        qa_types = (
            examples[type_field]
            if use_type
            else [None] * len(examples[question_field])
        )

        for question, answer, qa_type in zip(
            examples[question_field],
            examples[answer_field],
            qa_types,
        ):
            prompt = _build_prompt(
                tokenizer,
                str(question),
                str(qa_type) if qa_type is not None else None,
                system_prompt,
            )
            response = _build_response(tokenizer, str(answer))

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            if len(prompt_ids) >= max_length:
                input_ids = prompt_ids[:max_length]
                labels = [-100] * len(input_ids)
            else:
                response_budget = max_length - len(prompt_ids)
                response_ids = response_ids[:response_budget]
                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids

            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    return raw_splits.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing QA pairs",
    )
