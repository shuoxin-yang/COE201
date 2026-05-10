from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from utils.data import load_qa_dataset
from utils.log import Logger


def format_metrics(metrics: dict) -> dict:
    formatted = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 6)
        formatted[key] = value
    return formatted


def build_training_arguments(
    args: argparse.Namespace,
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
    }

    if has_eval_dataset:
        signature = inspect.signature(TrainingArguments).parameters
        eval_strategy_arg = (
            "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
        )
        training_args_kwargs[eval_strategy_arg] = "steps"
        training_args_kwargs["eval_steps"] = args.eval_steps

    return TrainingArguments(**training_args_kwargs)


class TrainerLoggerCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        metrics = {"global_step": state.global_step}
        metrics.update(format_metrics(logs))
        log_type = "Eval" if any(key.startswith("eval_") for key in logs) else "Train"
        self.logger.onlylog(
            metrics,
            name=f"Trainer {log_type} Log Step {state.global_step}",
        )
        self.logger.log_tensorboard(
            metrics,
            step=state.global_step,
            prefix=log_type.lower(),
        )
        self._log_loss_curves(logs, state.global_step)

    def _log_loss_curves(self, logs: dict, step: int) -> None:
        loss_metrics = {}
        if "loss" in logs:
            loss_metrics["train"] = logs["loss"]
        if "eval_loss" in logs:
            loss_metrics["test"] = logs["eval_loss"]
        if loss_metrics:
            self.logger.log_tensorboard(loss_metrics, step=step, prefix="loss")


class BestTestLossCheckpointCallback(TrainerCallback):
    def __init__(self, logger: Logger, tokenizer):
        self.logger = logger
        self.tokenizer = tokenizer
        self.best_loss: float | None = None
        self.best_step: int | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return

        test_loss = self._to_float(metrics["eval_loss"])
        if test_loss is None or not math.isfinite(test_loss):
            return
        if self.best_loss is not None and test_loss >= self.best_loss:
            return
        if not getattr(args, "should_save", True):
            return

        model = kwargs.get("model")
        if model is None:
            self.logger.onlylog(
                {"global_step": state.global_step, "eval_loss": test_loss},
                name="Best Test Loss Checkpoint Skipped",
            )
            return

        self.best_loss = test_loss
        self.best_step = state.global_step
        self._save_best_checkpoint(model, state, metrics, test_loss)
        self.logger.logandprint(
            {
                "global_step": self.best_step,
                "eval_loss": round(self.best_loss, 6),
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
            "eval_loss": test_loss,
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


def argparser():

    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument(
        "--model_name", type=str, default="Qwen3.5-4B", help="Pre-trained model path"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        help="Name for the current run, empty is 'lora_r={rank}_alpha={alpha}_scaling={scaling_type}'",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--alpha", type=float, default=16, help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--scaling_type",
        choices=["r/a", "r/sqrta"],
        default="r/a",
        help="LoRA scaling type",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="../QA_Gen/QA_pairs",
        help="Path to QA json/jsonl training data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum token length after Qwen chat-template formatting",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation split ratio in [0, 1); use 0 to train without eval split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/test split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help="Root directory for run logs and checkpoints",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default="~/autodl-tf",
        help="Root directory for TensorBoard event files; disabled when unset",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before optimizer step",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every N update steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="Evaluate test loss every N update steps when a test split exists",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoints every N update steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    return parser.parse_args()


def main():
    args = argparser()
    run_name = (
        args.run_name
        if args.run_name != ""
        else f"lora_r={args.rank}_alpha={args.alpha}_scaling={args.scaling_type}"
    )
    logger = Logger(
        log_path=args.output_dir,
        run_name=run_name,
        tensorboard_logdir=args.tensorboard_logdir,
    )
    assert (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ), "This script requires a GPU with bfloat16 support."
    device = torch.device("cuda")
    
    # Save config
    config = vars(args).copy()
    config["run_dir"] = logger.run_dir
    config["log_file"] = logger.log_file
    config["checkpoint_dir"] = logger.checkpoint_dir
    config["final_model_dir"] = logger.final_model_dir
    config["best_model_dir"] = logger.best_model_dir
    config["tensorboard_logdir"] = logger.tensorboard_logdir
    logger.onlylog(config, name="Config")
    
    # Load tokenizer and model
    
    # Tokenizer
    if os.path.isdir(args.model_name):
        model_path = args.model_name
    else:
        model_path = f"{os.path.abspath('.')}/../model/{args.model_name}"
        assert os.path.isdir(
            model_path
        ), f"Model directory {model_path} does not exist."
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.logandprint(
        f"Tokenizer native EOS ID: {tokenizer.eos_token_id} ({tokenizer.eos_token})"
    )
    
    # Model
    assert (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ), "This script requires a GPU with bfloat16 support."
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(
        device
    )
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=(args.scaling_type == "r/sqrta"),
    )
    model = get_peft_model(model, lora_cfg).to(device)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.logandprint(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    # Load dataset and preprocess it using the tokenizer
    dataset = load_qa_dataset(
        path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
    )
    eval_dataset = dataset["test"] if "test" in dataset else None
    logger.logandprint(f"train samples: {len(dataset['train'])}")
    if eval_dataset is not None:
        logger.logandprint(f"test samples: {len(eval_dataset)}")
    training_args = build_training_arguments(
        args=args,
        logger=logger,
        has_eval_dataset=eval_dataset is not None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    callbacks = [TrainerLoggerCallback(logger)]
    if eval_dataset is not None:
        callbacks.append(BestTestLossCheckpointCallback(logger, tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(dataset["train"])
    train_metrics["global_step"] = train_result.global_step
    train_metrics["training_loss"] = train_result.training_loss
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    formatted_train_metrics = format_metrics(train_metrics)
    logger.logandprint(formatted_train_metrics, name="Train Metrics")
    logger.log_tensorboard(formatted_train_metrics, prefix="train")
    logger.log_tensorboard(
        {"train": formatted_train_metrics["training_loss"]},
        step=train_result.global_step,
        prefix="loss",
    )

    trainer.save_model(logger.final_model_dir)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        eval_metrics["global_step"] = trainer.state.global_step
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        formatted_eval_metrics = format_metrics(eval_metrics)
        logger.logandprint(formatted_eval_metrics, name="Eval Metrics")
        logger.log_tensorboard(
            formatted_eval_metrics,
            step=trainer.state.global_step,
            prefix="eval",
        )
        logger.log_tensorboard(
            {"test": formatted_eval_metrics["eval_loss"]},
            step=trainer.state.global_step,
            prefix="loss",
        )
    else:
        logger.logandprint("eval skipped: no test split was created.")

    logger.close()


if __name__ == "__main__":
    main()
