import os
import sys
import json
import re
import glob
import hashlib
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HOME"] = str((Path(__file__).parent.parent / "model" / ".hf_cache").resolve())

import numpy as np
from tqdm import tqdm
import jieba

jieba.dt.tmp_dir = str(Path(__file__).parent.parent / "model")

from sentence_transformers import SentenceTransformer

CHINESE_STOP_WORDS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "为什么", "哪个", "哪些", "谁", "何时", "何地",
    "吗", "吧", "呢", "啊", "哦", "嗯", "嘛", "呀", "么", "的", "地", "得",
    "了", "着", "过", "把", "被", "让", "给", "对", "对于", "关于", "根据",
    "按照", "通过", "经过", "由于", "因为", "所以", "因此", "于是", "然后",
    "接着", "此外", "另外", "同时", "以及", "及其", "或是", "还是", "或者",
    "不是", "就是", "而是", "但是", "然而", "虽然", "尽管", "即使", "如果",
    "假如", "假设", "比如", "例如", "譬如", "之", "以", "而", "且", "与",
    "同", "较", "相比", "而言", "来说", "看", "来", "去", "做", "为", "等",
    "等等", "包括", "包括", "其中", "之间", "之内", "之外", "以后", "以前",
    "当前", "目前", "现在", "将来", "未来", "过去", "已经", "曾经", "刚刚",
    "正在", "将要", "可以", "能够", "应该", "必须", "需要", "可能", "也许",
    "大概", "大约", "左右", "前后", "上下", "多个", "一些", "一点", "有些",
    "许多", "很多", "大量", "少量", "这个", "那个", "这些", "那些", "每个",
    "任何", "所有", "全部", "一切", "有的", "某些",
}


class CFG:
    PER_TYPE_THRESHOLDS: Dict[str, Dict[str, float]] = {}

    @classmethod
    def get_qa_lower(cls, type_name: str) -> float:
        if type_name in cls.PER_TYPE_THRESHOLDS:
            return cls.PER_TYPE_THRESHOLDS[type_name].get("qa_lower", 0.5)
        return 0.5

    @classmethod
    def get_qq_upper(cls, type_name: str) -> float:
        if type_name in cls.PER_TYPE_THRESHOLDS:
            return cls.PER_TYPE_THRESHOLDS[type_name].get("qq_upper", 0.85)
        return 0.85

    @classmethod
    def prompt_per_type(cls, type_names: List[str]):
        print("\n" + "=" * 60)
        print("  Per-Type Threshold Configuration")
        print("=" * 60)
        for t in type_names:
            print(f"\n  --- {t} ---")
            while True:
                raw = input(f"  cos<Q,A> lower bound (default 0.5, < this => 答非所问): ").strip()
                if raw == "":
                    qa_val = 0.5
                    break
                try:
                    val = float(raw)
                    if 0.0 <= val <= 1.0:
                        qa_val = val
                        break
                    print("    Must be between 0 and 1.")
                except ValueError:
                    print("    Invalid input.")
            while True:
                raw = input(f"  cos<Q,Q> upper bound (default 0.85, > this => 重复): ").strip()
                if raw == "":
                    qq_val = 0.85
                    break
                try:
                    val = float(raw)
                    if 0.0 <= val <= 1.0:
                        qq_val = val
                        break
                    print("    Must be between 0 and 1.")
                except ValueError:
                    print("    Invalid input.")
            cls.PER_TYPE_THRESHOLDS[t] = {"qa_lower": qa_val, "qq_upper": qq_val}


def preprocess_text(text: str) -> str:
    text = re.sub(r'\\[.*?\\]|\\{.*?\\}|\\(.*?\\)', '', text)
    text = re.sub(r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a\u0030-\u0039]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = jieba.lcut(text)
    tokens = [t for t in tokens if t.strip() and t not in CHINESE_STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def validate_qa_pair(pair: dict) -> Tuple[bool, str]:
    if not isinstance(pair, dict):
        return False, "Data is not a dict"
    if "question" not in pair or "answer" not in pair:
        return False, "Missing 'question' or 'answer' field"
    if not isinstance(pair["question"], str) or not pair["question"].strip():
        return False, "Invalid or empty 'question'"
    if not isinstance(pair["answer"], str) or not pair["answer"].strip():
        return False, "Invalid or empty 'answer'"
    return True, ""


def compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class QAPairReader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def find_jsonl_files(self) -> List[Path]:
        pattern = "**/*.jsonl"
        files = sorted(self.data_dir.glob(pattern))
        return [f for f in files if f.is_file()]

    def read_all(self, show_progress: bool = True) -> List[dict]:
        files = self.find_jsonl_files()
        if not files:
            print(f"Warning: No JSONL files found in {self.data_dir}")
            return []

        all_pairs: List[dict] = []
        skipped_count = 0

        for file_path in tqdm(files, desc="Loading files", disable=not show_progress):
            try:
                pairs = self._read_single_file(file_path, show_progress)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"\nError reading {file_path}: {e}")
                skipped_count += 1

        print(f"Loaded {len(all_pairs)} QA pairs from {len(files) - skipped_count} files")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files due to errors")

        return all_pairs

    def _read_single_file(self, file_path: Path, show_progress: bool) -> List[dict]:
        pairs: List[dict] = []
        skipped = 0
        file_tag = file_path.stem.split(" ")[0]

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            line_iter = tqdm(lines, desc=f"  Reading {file_path.name}", leave=False) if show_progress else lines

            for line in line_iter:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    valid, _ = validate_qa_pair(data)
                    if valid:
                        t = data.get("type", "").strip()
                        pairs.append({
                            "type": t,
                            "question": data["question"].strip(),
                            "answer": data["answer"].strip(),
                            "_file": file_tag,
                        })
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1

        if skipped > 0 and show_progress:
            print(f"  ({file_path.name}: skipped {skipped} invalid lines)")

        return pairs


class SimilarityCalculator:
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        local_model_path = Path(__file__).parent.parent / "model" / "bge-base-zh-v1.5"
        if local_model_path.exists():
            print(f"Loading local BGE model from {local_model_path} on {device}...")
            self.model = SentenceTransformer(str(local_model_path), device=device)
        else:
            print(f"Downloading BGE model '{model_name}' and saving to {local_model_path}...")
            self.model = SentenceTransformer(model_name, device=device)
            self.model.save(str(local_model_path))
            print(f"Model saved to {local_model_path}")

        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._qq_cache: Dict[str, float] = {}
        self._qa_cache: Dict[str, float] = {}

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        uncached: List[Tuple[int, str]] = []
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        for i, text in enumerate(texts):
            text_hash = compute_content_hash(text)
            if text_hash in self._embedding_cache:
                results[i] = self._embedding_cache[text_hash]
            else:
                uncached.append((i, text))

        if uncached:
            batch_texts = [t for _, t in uncached]
            batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            for (idx, text), emb in zip(uncached, batch_embeddings):
                text_hash = compute_content_hash(text)
                self._embedding_cache[text_hash] = emb
                results[idx] = emb

        return results

    def compute_qq_similarity_matrix(self, questions: List[str]) -> np.ndarray:
        n = len(questions)
        if n == 0:
            return np.array([])
        embeddings = self.get_embeddings_batch(questions)
        sim_matrix = np.eye(n, dtype=np.float64)
        total = n * (n - 1) // 2
        with tqdm(total=total, desc="  QQ similarity") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    key = f"{compute_content_hash(questions[i])}:{compute_content_hash(questions[j])}"
                    if key in self._qq_cache:
                        sim = self._qq_cache[key]
                    else:
                        sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                        self._qq_cache[key] = sim
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
                    pbar.update(1)
        return sim_matrix

    def compute_qa_similarities(self, questions: List[str], answers: List[str]) -> List[float]:
        if len(questions) != len(answers):
            raise ValueError("questions and answers must have the same length")
        n = len(questions)
        results: List[float] = [0.0] * n
        for i in range(n):
            key = compute_content_hash(questions[i] + "|||" + answers[i])
            if key in self._qa_cache:
                results[i] = self._qa_cache[key]

        uncached = [i for i in range(n) if results[i] == 0.0]
        if uncached:
            q_texts = [questions[i] for i in uncached]
            a_texts = [answers[i] for i in uncached]
            q_embs = self.get_embeddings_batch(q_texts)
            a_embs = self.get_embeddings_batch(a_texts)
            with tqdm(total=len(uncached), desc="  QA similarity") as pbar:
                for idx_in_batch, orig_idx in enumerate(uncached):
                    sim = compute_cosine_similarity(q_embs[idx_in_batch], a_embs[idx_in_batch])
                    key = compute_content_hash(questions[orig_idx] + "|||" + answers[orig_idx])
                    self._qa_cache[key] = sim
                    results[orig_idx] = sim
                    pbar.update(1)
        return results


def compute_0p05_interval_stats(values: List[float]) -> List[Dict]:
    total = len(values)
    stats = []
    for i in range(20):
        low = round(i * 0.05, 2)
        high = round((i + 1) * 0.05, 2)
        if i == 19:
            count = sum(1 for v in values if low <= v <= high)
        else:
            count = sum(1 for v in values if low <= v < high)
        pct = (count / total * 100) if total > 0 else 0.0
        stats.append({"interval": f"{low:.2f} ~ {high:.2f}", "count": count, "percentage": pct})
    return stats


def compute_length_interval_stats(lengths: List[int], bin_size: int = 50) -> List[Dict]:
    if not lengths:
        return []
    total = len(lengths)
    max_val = max(lengths)
    max_regular_bin = min(max_val, bin_size * 12)
    stats = []
    for low in range(0, max_regular_bin, bin_size):
        high = low + bin_size
        count = sum(1 for v in lengths if low <= v < high)
        pct = (count / total * 100) if total > 0 else 0.0
        stats.append({"interval": f"{low} ~ {high}", "count": count, "percentage": pct})
    tail_count = sum(1 for v in lengths if v >= max_regular_bin)
    if tail_count > 0:
        pct = (tail_count / total * 100) if total > 0 else 0.0
        stats.append({"interval": f">= {max_regular_bin}", "count": tail_count, "percentage": pct})
    return stats


def display_table(title: str, columns: List[str], col_width: int, rows: List[List]):
    print(f"\n{'=' * (len(columns) * (col_width + 1) + 4)}")
    print(f"  {title}")
    print(f"{'=' * (len(columns) * (col_width + 1) + 4)}")
    header = "  "
    for c in columns:
        header += f" {c:<{col_width}}"
    print(header)
    print(f"  {'-' * (len(columns) * (col_width + 1) + 2)}")
    for row in rows:
        line = "  "
        for i, val in enumerate(row):
            line += f" {str(val):<{col_width}}"
        print(line)
    print(f"{'=' * (len(columns) * (col_width + 1) + 4)}")


def save_csv(file_path: Path, per_type_stats: Dict[str, List[Dict]]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    intervals = [s["interval"] for s in next(iter(per_type_stats.values()))]
    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["Interval"] + list(per_type_stats.keys())
        writer.writerow(header)
        for idx, interval in enumerate(intervals):
            row = [interval]
            for t in per_type_stats:
                row.append(round(per_type_stats[t][idx]["percentage"], 2))
            writer.writerow(row)
    print(f"  Saved to: {file_path.resolve()}")


class FilterEngine:
    def __init__(self, qa_pairs: List[dict], qq_matrix: np.ndarray, qa_sims: List[float]):
        self.qa_pairs = qa_pairs
        self.qq_matrix = qq_matrix
        self.qa_sims = qa_sims

    def filter_by_qq(self, qq_upper: float) -> Tuple[List[int], List[int], List[Tuple[int, int, float]]]:
        n = self.qq_matrix.shape[0]
        duplicate_pairs = []
        kept = set(range(n))
        for i in range(n):
            if i not in kept:
                continue
            for j in range(i + 1, n):
                if j not in kept:
                    continue
                sim = self.qq_matrix[i, j]
                if sim > qq_upper:
                    duplicate_pairs.append((i, j, sim))
                    kept.discard(j)
        kept_list = sorted(kept)
        removed_list = [i for i in range(n) if i not in kept]
        return kept_list, removed_list, duplicate_pairs

    def run(self, qq_upper: float, qa_lower: float) -> List[dict]:
        qq_kept, qq_removed, dup_pairs = self.filter_by_qq(qq_upper)
        qa_kept = []
        qa_removed = []
        for i in qq_kept:
            if self.qa_sims[i] >= qa_lower:
                qa_kept.append(i)
            else:
                qa_removed.append(i)
        return [self.qa_pairs[i] for i in qa_kept]


def show_file_distribution(all_pairs: List[dict], title: str):
    files = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    counts = {f: 0 for f in files}
    for p in all_pairs:
        f = p.get("_file", "??")
        if f in counts:
            counts[f] += 1
    print(f"\n  {title}")
    print(f"  {'Chapter':<10} {'Count':<8}")
    print(f"  {'-' * 18}")
    total = 0
    for f in files:
        c = counts.get(f, 0)
        print(f"  {f:<10} {c:<8}")
        total += c
    print(f"  {'-' * 18}")
    print(f"  {'Total':<10} {total:<8}")


def save_filtered_pairs(pairs: List[dict]):
    output_path = Path(__file__).parent/ "data" / "filtered_output.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            out = {"question": pair["question"], "answer": pair["answer"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\nFiltered QA pairs saved to: {output_path.resolve()}")
    print(f"Total pairs in output: {len(pairs)}")


def show_length_stats_and_save(all_pairs_by_type: Dict[str, List[dict]], csv_dir: Path):
    print(f"\n{'=' * 70}")
    print("  Token Length Statistics (character count)")
    print(f"{'=' * 70}")

    type_names = list(all_pairs_by_type.keys())

    q_len_by_type: Dict[str, List[int]] = {}
    a_len_by_type: Dict[str, List[int]] = {}
    qa_max_len_by_type: Dict[str, List[int]] = {}
    all_q_lens: List[int] = []
    all_a_lens: List[int] = []
    all_qa_lens: List[int] = []

    for t, pairs in all_pairs_by_type.items():
        q_lens = [len(p["question"]) for p in pairs]
        a_lens = [len(p["answer"]) for p in pairs]
        qa_lens = [max(len(p["question"]), len(p["answer"])) for p in pairs]
        q_len_by_type[t] = q_lens
        a_len_by_type[t] = a_lens
        qa_max_len_by_type[t] = qa_lens
        all_q_lens.extend(q_lens)
        all_a_lens.extend(a_lens)
        all_qa_lens.extend(qa_lens)

    qa_stats: Dict[str, List[Dict]] = {}
    q_stats: Dict[str, List[Dict]] = {}
    a_stats: Dict[str, List[Dict]] = {}
    for t in type_names:
        qa_stats[t] = compute_length_interval_stats(qa_max_len_by_type[t])
        q_stats[t] = compute_length_interval_stats(q_len_by_type[t])
        a_stats[t] = compute_length_interval_stats(a_len_by_type[t])
    qa_stats["全部"] = compute_length_interval_stats(all_qa_lens)
    q_stats["全部"] = compute_length_interval_stats(all_q_lens)
    a_stats["全部"] = compute_length_interval_stats(all_a_lens)

    all_columns = type_names + ["全部"]

    for caption, stats_dict in [("QA pair max(Q_len, A_len) distribution", qa_stats),
                                 ("Q length distribution", q_stats),
                                 ("A length distribution", a_stats)]:
        intervals = [s["interval"] for s in next(iter(stats_dict.values()))]
        rows = []
        for idx, interval in enumerate(intervals):
            row = [interval]
            for t in all_columns:
                pct = stats_dict[t][idx]["percentage"]
                row.append(f"{pct:.2f}%")
            rows.append(row)
        display_table(caption, ["Interval"] + all_columns, 16, rows)

    save_csv(csv_dir / "QA_pair_length_distribution.csv", qa_stats)
    save_csv(csv_dir / "Q_length_distribution.csv", q_stats)
    save_csv(csv_dir / "A_length_distribution.csv", a_stats)

    print(f"\n  {'Type':<20} {'QA_pair_maxlen':<18} {'Q_maxlen':<15} {'A_maxlen':<15}")
    print(f"  {'-' * 68}")
    for t in type_names:
        qa_max = max(qa_max_len_by_type[t]) if qa_max_len_by_type[t] else 0
        q_max = max(q_len_by_type[t]) if q_len_by_type[t] else 0
        a_max = max(a_len_by_type[t]) if a_len_by_type[t] else 0
        print(f"  {t:<20} {qa_max:<18} {q_max:<15} {a_max:<15}")
    qa_max_all = max(all_qa_lens) if all_qa_lens else 0
    q_max_all = max(all_q_lens) if all_q_lens else 0
    a_max_all = max(all_a_lens) if all_a_lens else 0
    print(f"  {'全部':<20} {qa_max_all:<18} {q_max_all:<15} {a_max_all:<15}")
    print()


def prompt_global_length_filter(all_pairs_by_type: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    print("\n" + "=" * 60)
    print("  Global Maxlength Filtering")
    print("  (Applies to ALL types uniformly)")
    print("=" * 60)

    qa_max_q = input("  Set QA pair maxlength (max(Q_len,A_len)) [default=skip]: ").strip()
    if qa_max_q == "":
        qa_limit = None
    else:
        try:
            val = int(qa_max_q)
            if val > 0:
                qa_limit = val
            else:
                print("    Must be positive, skipping.")
                qa_limit = None
        except ValueError:
            print("    Invalid, skipping.")
            qa_limit = None

    q_max_q = input("  Set Q maxlength [default=skip]: ").strip()
    if q_max_q == "":
        q_limit = None
    else:
        try:
            val = int(q_max_q)
            if val > 0:
                q_limit = val
            else:
                print("    Must be positive, skipping.")
                q_limit = None
        except ValueError:
            print("    Invalid, skipping.")
            q_limit = None

    a_max_q = input("  Set A maxlength [default=skip]: ").strip()
    if a_max_q == "":
        a_limit = None
    else:
        try:
            val = int(a_max_q)
            if val > 0:
                a_limit = val
            else:
                print("    Must be positive, skipping.")
                a_limit = None
        except ValueError:
            print("    Invalid, skipping.")
            a_limit = None

    if qa_limit is None and q_limit is None and a_limit is None:
        return all_pairs_by_type

    result: Dict[str, List[dict]] = {}
    total_before = 0
    total_after = 0
    for t, pairs in all_pairs_by_type.items():
        before = len(pairs)
        total_before += before
        after = []
        for p in pairs:
            q_len = len(p["question"])
            a_len = len(p["answer"])
            pair_max = max(q_len, a_len)
            if qa_limit is not None and pair_max > qa_limit:
                continue
            if q_limit is not None and q_len > q_limit:
                continue
            if a_limit is not None and a_len > a_limit:
                continue
            after.append(p)
        result[t] = after
        total_after += len(after)
        print(f"  {t}: {before} -> {len(after)} (removed {before - len(after)})")

    print(f"\n  Total: {total_before} -> {total_after} (removed {total_before - total_after})")

    confirm = input("  Apply this length filter? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Length filter discarded, keeping original data.")
        return all_pairs_by_type

    return result


def main():
    print("=" * 60)
    print("         QA Pair Filter - Data Filtering Tool")
    print("=" * 60)

    data_dir = Path(__file__).parent / "data" / "RawQA"
    print(f"Data directory: {data_dir}")

    reader = QAPairReader(str(data_dir))
    all_pairs = reader.read_all(show_progress=True)
    if not all_pairs:
        print("No QA pairs found. Exiting.")
        return

    type_names = sorted(set(p["type"] for p in all_pairs if p.get("type")))
    print(f"\nDetected types: {', '.join(type_names)}")
    print(f"Total pairs: {len(all_pairs)}")

    pairs_by_type: Dict[str, List[dict]] = {t: [] for t in type_names}
    for p in all_pairs:
        pairs_by_type[p["type"]].append(p)

    for t in type_names:
        print(f"  {t}: {len(pairs_by_type[t])} pairs")

    for t in type_names:
        show_file_distribution(pairs_by_type[t], f"Distribution of {t} across chapters")

    csv_dir = Path(__file__).parent / "data_mformation"
    show_length_stats_and_save(pairs_by_type, csv_dir)

    pairs_by_type = prompt_global_length_filter(pairs_by_type)
    all_pairs_after_length = []
    for t in type_names:
        all_pairs_after_length.extend(pairs_by_type[t])
    print(f"\nTotal pairs after length filtering: {len(all_pairs_after_length)}")

    calculator = SimilarityCalculator()

    qq_per_type: Dict[str, List[float]] = {}
    qa_per_type: Dict[str, List[float]] = {}

    for t in type_names:
        t_pairs = pairs_by_type[t]
        if len(t_pairs) < 2:
            print(f"\n  {t}: only {len(t_pairs)} pair(s), skipping QQ matrix")
            qq_per_type[t] = []
            processed_qs = [preprocess_text(p["question"]) for p in t_pairs]
            processed_as = [preprocess_text(p["answer"]) for p in t_pairs]
            qa_sims = calculator.compute_qa_similarities(processed_qs, processed_as)
            qa_per_type[t] = qa_sims
            continue

        print(f"\n  === Processing type: {t} ({len(t_pairs)} pairs) ===")
        questions = [p["question"] for p in t_pairs]
        answers = [p["answer"] for p in t_pairs]

        processed_qs = []
        for q in tqdm(questions, desc=f"  Preprocessing Q"):
            processed_qs.append(preprocess_text(q))
        processed_as = []
        for a in tqdm(answers, desc=f"  Preprocessing A"):
            processed_as.append(preprocess_text(a))

        qa_sims = calculator.compute_qa_similarities(processed_qs, processed_as)
        qa_per_type[t] = qa_sims

        qq_matrix = calculator.compute_qq_similarity_matrix(processed_qs)
        n = len(questions)
        qq_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                qq_vals.append(qq_matrix[i, j])
        qq_per_type[t] = qq_vals

        print(f"  {t}: QA computed, QQ computed ({len(qq_vals)} pairs)")

    qq_stats_per_type: Dict[str, List[Dict]] = {}
    qa_stats_per_type: Dict[str, List[Dict]] = {}
    for t in type_names:
        qq_stats_per_type[t] = compute_0p05_interval_stats(qq_per_type.get(t, []))
        qa_stats_per_type[t] = compute_0p05_interval_stats(qa_per_type.get(t, []))

    all_columns = type_names
    intervals_qq = [s["interval"] for s in next(iter(qq_stats_per_type.values()))]
    rows_qq = []
    for idx, interval in enumerate(intervals_qq):
        row = [interval]
        for t in all_columns:
            pct = qq_stats_per_type[t][idx]["percentage"]
            row.append(f"{pct:.2f}%")
        rows_qq.append(row)
    display_table("QQ Similarity Distribution (per type)", ["Interval"] + all_columns, 16, rows_qq)

    intervals_qa = [s["interval"] for s in next(iter(qa_stats_per_type.values()))]
    rows_qa = []
    for idx, interval in enumerate(intervals_qa):
        row = [interval]
        for t in all_columns:
            pct = qa_stats_per_type[t][idx]["percentage"]
            row.append(f"{pct:.2f}%")
        rows_qa.append(row)
    display_table("QA Similarity Distribution (per type)", ["Interval"] + all_columns, 16, rows_qa)

    save_csv(csv_dir / "QQ_distribution.csv", qq_stats_per_type)
    save_csv(csv_dir / "QA_distribution.csv", qa_stats_per_type)

    while True:
        CFG.prompt_per_type(type_names)

        for t in type_names:
            t_pairs = pairs_by_type[t]
            qq_u = CFG.get_qq_upper(t)
            qa_l = CFG.get_qa_lower(t)
            print(f"\n  --- {t}: cos<Q,Q> > {qq_u} => 重复, cos<Q,A> < {qa_l} => 答非所问 ---")

            if len(t_pairs) < 2:
                engine = FilterEngine(t_pairs, np.eye(1), qa_per_type.get(t, []))
                filtered = engine.run(qq_u, qa_l)
            else:
                questions = [p["question"] for p in t_pairs]
                processed_qs = [preprocess_text(q) for q in questions]
                qq_matrix = calculator.compute_qq_similarity_matrix(processed_qs)
                engine = FilterEngine(t_pairs, qq_matrix, qa_per_type[t])
                filtered = engine.run(qq_u, qa_l)

            pairs_by_type[t] = filtered
            print(f"    Result: {len(filtered)} / {len(t_pairs)} kept")
            show_file_distribution(filtered, f"Filtered {t} distribution")

        confirm = input("\nAccept QQ/QA filtering results? (Y/n): ").strip().lower()
        if confirm == "n":
            print("\nRe-enter boundary conditions.\n")
            continue
        break

    all_filtered = []
    for t in type_names:
        all_filtered.extend(pairs_by_type[t])

    save_filtered_pairs(all_filtered)
    print("\nDone. Thank you for using QA Pair Filter!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
