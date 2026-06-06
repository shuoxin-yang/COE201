import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAIN_DIR.parent
QA_GEN_DIR = PROJECT_DIR / "QA_Gen"

PROCESSED_QA_DIR = QA_GEN_DIR / "data" / "ProcessedQA"
QA_AUGMENTATION_DIR = QA_GEN_DIR / "data" / "QA_Augmentation"
DEFAULT_OUTPUT_DIR = TRAIN_DIR / "logs" / "QA_cos"

# QA_Filter.py expects Project/model to exist for jieba/HF caches.
(PROJECT_DIR / "model").mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(QA_GEN_DIR))

from QA_Filter import (  # noqa: E402
    SimilarityCalculator,
    compute_cosine_similarity,
    preprocess_text,
    validate_qa_pair,
)


def stable_file_seed(seed: int, file_path: Path) -> int:
    digest = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def read_qa_pairs(file_path: Path) -> Tuple[List[dict], int]:
    pairs: List[dict] = []
    skipped = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            valid, _ = validate_qa_pair(data)
            if not valid:
                skipped += 1
                continue

            pairs.append(
                {
                    "line_no": line_no,
                    "type": str(data.get("type", "")).strip(),
                    "question": data["question"].strip(),
                    "answer": data["answer"].strip(),
                }
            )

    return pairs, skipped


def sample_index_pairs(total_items: int, sample_size: int, rng: random.Random) -> List[Tuple[int, int]]:
    if total_items < 2 or sample_size <= 0:
        return []

    total_combinations = total_items * (total_items - 1) // 2
    if total_combinations <= sample_size:
        return [(i, j) for i in range(total_items) for j in range(i + 1, total_items)]

    sampled = set()
    while len(sampled) < sample_size:
        i = rng.randrange(total_items)
        j = rng.randrange(total_items - 1)
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        sampled.add((i, j))

    return sorted(sampled)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "max": float(np.max(arr)),
    }


def write_detail_csv(output_path: Path, rows: Sequence[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "file",
        "sample_no",
        "line_no_a",
        "line_no_b",
        "type_a",
        "type_b",
        "cosine_similarity",
        "question_a",
        "question_b",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(output_path: Path, rows: Sequence[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "file",
        "valid_qa_count",
        "skipped_line_count",
        "sampled_pair_count",
        "mean",
        "std",
        "min",
        "p25",
        "median",
        "p75",
        "max",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_file_cosines(
    dataset_name: str,
    file_path: Path,
    sample_size: int,
    seed: int,
    calculator: SimilarityCalculator,
) -> Tuple[List[dict], dict]:
    pairs, skipped = read_qa_pairs(file_path)
    rng = random.Random(stable_file_seed(seed, file_path))
    index_pairs = sample_index_pairs(len(pairs), sample_size, rng)

    if not index_pairs:
        summary = {
            "dataset": dataset_name,
            "file": file_path.name,
            "valid_qa_count": len(pairs),
            "skipped_line_count": skipped,
            "sampled_pair_count": 0,
            **summarize([]),
        }
        return [], summary

    needed_indices = sorted({idx for pair in index_pairs for idx in pair})
    processed_questions = [preprocess_text(pairs[idx]["question"]) for idx in needed_indices]
    embeddings = calculator.get_embeddings_batch(processed_questions)
    embedding_by_index = dict(zip(needed_indices, embeddings))

    detail_rows: List[dict] = []
    similarities: List[float] = []
    for sample_no, (i, j) in enumerate(index_pairs, start=1):
        sim = compute_cosine_similarity(embedding_by_index[i], embedding_by_index[j])
        similarities.append(sim)
        detail_rows.append(
            {
                "dataset": dataset_name,
                "file": file_path.name,
                "sample_no": sample_no,
                "line_no_a": pairs[i]["line_no"],
                "line_no_b": pairs[j]["line_no"],
                "type_a": pairs[i]["type"],
                "type_b": pairs[j]["type"],
                "cosine_similarity": f"{sim:.8f}",
                "question_a": pairs[i]["question"],
                "question_b": pairs[j]["question"],
            }
        )

    summary = {
        "dataset": dataset_name,
        "file": file_path.name,
        "valid_qa_count": len(pairs),
        "skipped_line_count": skipped,
        "sampled_pair_count": len(index_pairs),
        **{k: f"{v:.8f}" for k, v in summarize(similarities).items()},
    }
    return detail_rows, summary


def iter_dataset_files(dataset_dirs: Sequence[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for dataset_name, dataset_dir in dataset_dirs:
        for file_path in sorted(dataset_dir.glob("*.jsonl")):
            if file_path.is_file():
                files.append((dataset_name, file_path))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample QA-pair pairs per file and compute question-question cosine similarity."
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="number of QA-pair pairs sampled per file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", default=None, help="sentence-transformers device, e.g. cpu/cuda/mps")
    parser.add_argument("--model-name", default="BAAI/bge-base-zh-v1.5", help="sentence-transformers model name")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_QA_DIR)
    parser.add_argument("--augmentation-dir", type=Path, default=QA_AUGMENTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = [
        ("ProcessedQA", args.processed_dir),
        ("QA_Augmentation", args.augmentation_dir),
    ]

    missing_dirs = [str(path) for _, path in dataset_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing data directories: {', '.join(missing_dirs)}")

    dataset_files = iter_dataset_files(dataset_dirs)
    if not dataset_files:
        raise FileNotFoundError("No .jsonl files found in the configured data directories.")

    print(f"Found {len(dataset_files)} JSONL files.")
    print(f"Sampling up to {args.sample_size} QA-pair pairs per file with seed {args.seed}.")
    print("Loading similarity model...")
    calculator = SimilarityCalculator(model_name=args.model_name, device=args.device)

    all_detail_rows: List[dict] = []
    summary_rows: List[dict] = []
    for dataset_name, file_path in tqdm(dataset_files, desc="Processing files"):
        detail_rows, summary = compute_file_cosines(
            dataset_name=dataset_name,
            file_path=file_path,
            sample_size=args.sample_size,
            seed=args.seed,
            calculator=calculator,
        )
        all_detail_rows.extend(detail_rows)
        summary_rows.append(summary)

        print(
            f"  {dataset_name}/{file_path.name}: "
            f"{summary['sampled_pair_count']} samples, mean={summary['mean']}"
        )

    detail_path = args.output_dir / "question_cosine_samples.csv"
    summary_path = args.output_dir / "question_cosine_summary.csv"
    write_detail_csv(detail_path, all_detail_rows)
    write_summary_csv(summary_path, summary_rows)

    print("\nDone.")
    print(f"Detail CSV:  {detail_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
