import os
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
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


def argparser():

    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument(
        "--model_name", type=str, default="Qwen3.5-4B", help="Pre-trained model path"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
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
        default=0.1,
        help="Validation split ratio in [0, 1); use 0 to train without eval split",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used for train/test split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_finetuned_model",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of update steps to accumulate before optimizer step",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning",
    )
    return parser.parse_args()


def main():
    args = argparser()
    logger = Logger(
        log_path="./logs",
        run_name=(
            args.run_name
            if args.run_name != ""
            else f"lora_r={args.rank}_alpha={args.alpha}_scaling={args.scaling_type}"
        ),
    )
    assert (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ), "This script requires a GPU with bfloat16 support."
    device = torch.device("cuda")
    
    # Save config
    logger.onlylog(vars(args), name="Config")
    
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
    )
    dataset = load_qa_dataset(
        path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    eval_dataset = dataset["test"] if "test" in dataset else None
    logger.logandprint(f"train samples: {len(dataset['train'])}")
    if eval_dataset is not None:
        logger.logandprint(f"test samples: {len(eval_dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
    logger.logandprint(format_metrics(train_metrics), name="Train Metrics")

    trainer.save_model(args.output_dir)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.logandprint(format_metrics(eval_metrics), name="Eval Metrics")
    else:
        logger.logandprint("eval skipped: no test split was created.")


if __name__ == "__main__":
    main()
