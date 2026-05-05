from __future__ import annotations

from typing import Any

from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizerBase


DEFAULT_QA_SYSTEM_PROMPT = ""


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
    system_prompt: str = DEFAULT_QA_SYSTEM_PROMPT,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question.strip()})

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


def _validate_test_size(test_size: float, dataset_len: int) -> None:
    if test_size < 0 or test_size >= 1:
        raise ValueError("--test_size must be in [0, 1). Use 0 to disable eval split.")
    if dataset_len < 2 and test_size > 0:
        raise ValueError("Need at least 2 samples when --test_size is greater than 0.")


def load_qa_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    test_size: float = 0.1,
    seed: int = 42,
    question_field: str = "question",
    answer_field: str = "answer",
    system_prompt: str = DEFAULT_QA_SYSTEM_PROMPT,
    data_format: str = "json",
) -> DatasetDict:
    """Load QA json/jsonl data and return tokenized train/test splits for Qwen SFT."""
    if path is None:
        raise ValueError("path must be provided.")
    if tokenizer is None:
        raise ValueError("tokenizer must be provided.")
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0.")

    raw_dataset = hf_load_dataset(data_format, data_files=path)["train"]
    _validate_qa_columns(raw_dataset.column_names, question_field, answer_field)
    _validate_test_size(test_size, len(raw_dataset))

    if test_size > 0:
        raw_splits = raw_dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            shuffle=True,
        )
    else:
        raw_splits = DatasetDict({"train": raw_dataset})

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for question, answer in zip(examples[question_field], examples[answer_field]):
            prompt = _build_prompt(tokenizer, str(question), system_prompt)
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
