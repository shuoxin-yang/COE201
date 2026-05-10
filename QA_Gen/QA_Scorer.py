import os
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_ROOT = PROJECT_ROOT / "model"
QA_PROCESSEDQA_DIR = Path(__file__).parent / "data" / "ProcessedQA"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 64


SCORE_SYSTEM_PROMPT = """你是一位严谨的离散数学QA质量评审专家。你的任务是对给定的【问题-答案】对进行质量评分。

请从准确性、完整性、逻辑性、表述清晰度、答案详细程度五个维度综合给出一个整数评分（0-5分），仅输出评分，不要输出任何解释。评分标准如下：

- 5分：完全正确，答案完整、清晰，与课程材料一致，可直接用于教学。
- 4分：基本正确，略有不足，但核心信息准确。
- 3分：部分正确，存在小错误，但不影响对核心概念的理解。
- 2分：错误较多，核心信息不准确。
- 1分：完全错误或答非所问。
- 0分：无效内容。

仅输出一个整数0-5，不要包含任何其他文字。"""


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        return str(model_path)
    project_model_path = MODEL_ROOT / model_name
    if project_model_path.is_dir():
        return str(project_model_path)
    return model_name


def select_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def setup_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_scoring_model(model_path: str):
    device, dtype = select_device_and_dtype()
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Loading scoring model from {model_path}...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    except Exception:
        print("4-bit quantization failed, trying to load with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    print("Scoring model loaded successfully.")
    return model


def build_scoring_prompt(question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": f"请评估以下离散数学QA对的质量：\n\n问题：{question}\n\n答案：{answer}"},
    ]
    return messages


def strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, count=1, flags=re.DOTALL).strip()


def extract_score(text: str) -> Optional[int]:
    text = strip_think(text).strip()
    for pattern in [
        r'"score"\s*:\s*(\d)',
        r'score["\s:=]+\s*(\d)',
        r'得分为?\s*(\d)',
        r'评分[为:：]?\s*(\d)',
        r'^(\d)$',
        r'\b([0-5])\b',
    ]:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 5:
                return score
    return None


def score_qa_pair(model, tokenizer, question: str, answer: str, debug: bool = False) -> Dict:
    messages = build_scoring_prompt(question, answer)
    for attempt in range(3):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                )

            generated_ids = outputs[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            score = extract_score(response_text)
            if score is not None:
                return {"score": score}
            if debug:
                print(f"\n[DEBUG] Raw model output: '{response_text[:300]}'")
            if attempt < 2:
                time.sleep(1)
                continue
            if debug:
                print(f"  Failed to extract score. Raw: '{response_text[:300]}'")
            return {"score": 0}

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < 2:
                print("  OOM detected, clearing cache and retrying...")
                time.sleep(2)
                continue
            return {"score": 0}
        except Exception as e:
            if attempt < 2:
                print(f"  Error on attempt {attempt + 1}: {e}, retrying...")
                time.sleep(1)
                continue
            return {"score": 0}


def score_qa_pair_batch(model, tokenizer, questions: List[str], answers: List[str], max_retries: int = 2) -> List[Dict]:
    messages_list = [build_scoring_prompt(q, a) for q, a in zip(questions, answers)]
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for messages in messages_list
    ]
    for attempt in range(max_retries + 1):
        try:
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
            tokenizer.padding_side = original_padding_side
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                )

            results = []
            for i in range(len(texts)):
                pad_len = (inputs["attention_mask"][i] == 0).sum().item()
                input_len = (inputs["attention_mask"][i] == 1).sum().item()
                generated_ids = outputs[i][pad_len + input_len:]
                response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                score = extract_score(response_text)
                if score is not None:
                    results.append({"score": score})
                else:
                    results.append({"score": 0})
            return results

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < max_retries:
                print(f"  Batch OOM (size={len(texts)}), retrying...")
                time.sleep(2)
                continue
            return [{"score": 0} for _ in range(len(texts))]
        except Exception as e:
            if attempt < max_retries:
                print(f"  Batch error: {e}, retrying...")
                time.sleep(1)
                continue
            return [{"score": 0} for _ in range(len(texts))]


def read_filtered_pairs(file_path: Path) -> List[Dict]:
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "question" not in data or "answer" not in data:
                    print(f"  Warning: Line {line_num} missing 'question' or 'answer', skipping.")
                    continue
                pairs.append(data)
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} invalid JSON: {e}, skipping.")
    return pairs


def save_scored_pairs(pairs: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Scored QA pairs saved to: {output_path.resolve()}")
    print(f"Total pairs: {len(pairs)}")


def compute_score_distribution(pairs: List[Dict]) -> Dict[int, int]:
    distribution: Dict[int, int] = {}
    for p in pairs:
        score = p.get("score", 0)
        distribution[score] = distribution.get(score, 0) + 1
    return distribution


def display_score_statistics(pairs: List[Dict]):
    total = len(pairs)
    if total == 0:
        print("\nNo QA pairs to analyze.")
        return
    dist = compute_score_distribution(pairs)
    print("\n" + "=" * 60)
    print("  QA Pair Score Distribution")
    print("=" * 60)
    print(f"  {'Score':<8} {'Count':<8} {'Percentage':<12}")
    print(f"  {'-' * 28}")
    for score in sorted(dist.keys(), reverse=True):
        count = dist[score]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {score:<8} {count:<8} {pct:>6.2f}%   {bar}")
    print(f"  {'-' * 28}")
    print(f"  {'Total':<8} {total:<8} {'100.00%':<12}")
    print()

    qualified = sum(count for s, count in dist.items() if s >= 3)
    print(f"  Qualified (score >= 3): {qualified} / {total} ({qualified / total * 100:.2f}%)")
    print(f"  Unqualified (score < 3): {total - qualified} / {total} ({(total - qualified) / total * 100:.2f}%)")
    print("=" * 60)


def prompt_score_threshold() -> int:
    while True:
        raw = input("\nEnter score threshold (QA pairs with score < threshold will be removed): ").strip()
        try:
            threshold = int(raw)
            if 0 <= threshold <= 5:
                return threshold
            print("  Threshold must be between 0 and 5.")
        except ValueError:
            print("  Invalid input. Please enter an integer between 0 and 5.")


def filter_by_threshold(pairs: List[Dict], threshold: int) -> Tuple[List[Dict], List[Dict]]:
    kept = [p for p in pairs if p.get("score", 0) >= threshold]
    removed = [p for p in pairs if p.get("score", 0) < threshold]
    return kept, removed


def main():
    print("=" * 60)
    print("         QA Pair Scorer - Quality Scoring & Filtering")
    print("=" * 60)

    filtered_path = QA_PROCESSEDQA_DIR / "filtered_output.jsonl"
    if not filtered_path.exists():
        print(f"Error: {filtered_path} not found.")
        print("Please run QA_Filter.py first to generate filtered_output.jsonl.")
        sys.exit(1)

    print(f"Reading QA pairs from: {filtered_path}")
    all_pairs = read_filtered_pairs(filtered_path)
    if not all_pairs:
        print("No valid QA pairs found. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(all_pairs)} QA pairs.\n")

    model_name = "Qwen3.6-27B"
    model_path = resolve_model_path(model_name)
    print(f"Model path: {model_path}")
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at {model_path}")
        print("Please ensure the Qwen3.6-27B model is placed in the model/ directory.")
        sys.exit(1)

    tokenizer = setup_tokenizer(model_path)
    model = load_scoring_model(model_path)

    print(f"\nScoring {len(all_pairs)} QA pairs with batch_size={BATCH_SIZE}, enable_thinking=False...")
    scored_pairs = []
    scoring_errors = 0

    for idx in range(0, len(all_pairs), BATCH_SIZE):
        batch = all_pairs[idx:idx + BATCH_SIZE]
        valid_indices = []
        questions = []
        answers = []

        for i, pair in enumerate(batch):
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            if not question or not answer:
                pair["score"] = 0
                scoring_errors += 1
                scored_pairs.append(pair)
            else:
                valid_indices.append(i)
                questions.append(question)
                answers.append(answer)

        if questions:
            if len(questions) > 1:
                results = score_qa_pair_batch(model, tokenizer, questions, answers)
            else:
                debug = (idx == 0)
                results = [score_qa_pair(model, tokenizer, questions[0], answers[0], debug=debug)]

            for j, result in zip(valid_indices, results):
                pair = batch[j]
                pair["score"] = result["score"]
                if result["score"] == 0 and idx == 0:
                    pass
                scored_pairs.append(pair)

        if ((idx // BATCH_SIZE) + 1) % 5 == 0:
            torch.cuda.empty_cache()

        tqdm.write(f"  Processed {min(idx + BATCH_SIZE, len(all_pairs))}/{len(all_pairs)} pairs")

    print(f"\nScoring complete. Total: {len(scored_pairs)}, Errors/failed: {scoring_errors}")

    scored_path = QA_PROCESSEDQA_DIR / "scored_output.jsonl"
    save_scored_pairs(scored_pairs, scored_path)

    display_score_statistics(scored_pairs)

    threshold = prompt_score_threshold()
    kept_pairs, removed_pairs = filter_by_threshold(scored_pairs, threshold)
    print(f"\nFiltering results (threshold = {threshold}):")
    print(f"  Kept: {len(kept_pairs)} pairs (score >= {threshold})")
    print(f"  Removed: {len(removed_pairs)} pairs (score < {threshold})")

    filtered_scored_path = QA_PROCESSEDQA_DIR / "filtered_scored_output.jsonl"
    save_scored_pairs(kept_pairs, filtered_scored_path)

    print(f"\nAll done! Final output: {filtered_scored_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
