#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TRAIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = TRAIN_DIR / "logs"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from utils.log import build_lora_run_core_name, build_lora_run_dir_name


MODEL_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?B)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

MODULE_CODE_TO_TARGET = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename old LoRA log directories to the new timestamped format."
    )
    parser.add_argument(
        "log_root",
        nargs="?",
        default=str(DEFAULT_LOG_ROOT),
        help=f"Directory containing saved run directories. Default: {DEFAULT_LOG_ROOT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing directories.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Rename without asking for per-directory confirmation.",
    )
    parser.add_argument(
        "--include-all-dirs",
        action="store_true",
        help="Try every child directory, even if its name/log does not mention LoRA.",
    )
    parser.add_argument(
        "--infer-missing-layers",
        action="store_true",
        help="Use full when layer selection is missing instead of prompting.",
    )
    parser.add_argument(
        "--suffix-conflicts",
        action="store_true",
        help="Append __dupN when the target directory already exists.",
    )
    parser.add_argument(
        "--tensorboard-root",
        type=str,
        default=None,
        help=(
            "TensorBoard root containing run directories. By default this is "
            "inferred from log.txt or config.yaml."
        ),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Only rename saved log directories and leave TensorBoard logs untouched.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_root = Path(args.log_root).expanduser().resolve()
    if not log_root.is_dir():
        print(f"[error] log root does not exist: {log_root}")
        return 1

    converted = 0
    skipped = 0
    for run_dir in sorted(path for path in log_root.iterdir() if path.is_dir()):
        result = convert_run_dir(run_dir, args)
        if result:
            converted += 1
        else:
            skipped += 1

    action = "would convert" if args.dry_run else "converted"
    print(f"[done] {action}: {converted}, skipped: {skipped}")
    return 0


def convert_run_dir(run_dir: Path, args: argparse.Namespace) -> bool:
    if not args.include_all_dirs and not is_lora_run(run_dir):
        return False

    config = load_config(run_dir)
    if config is None:
        print(f"[skip] {run_dir.name}: missing config.yaml")
        return False

    config = copy.deepcopy(config)
    patch_missing_from_dir_name(config, run_dir.name)
    try:
        ensure_layer_selection(config, run_dir.name, args)
        target_name = build_lora_run_dir_name(
            config,
            run_name=infer_run_name(run_dir, config),
            timestamp=infer_run_timestamp(run_dir),
        )
    except ValueError as exc:
        print(f"[skip] {run_dir.name}: {exc}")
        return False

    if run_dir.name == target_name:
        print(f"[skip] {run_dir.name}: already uses the new name")
        return False

    tensorboard_dir = find_tensorboard_dir(run_dir, config, args)
    try:
        target_dir, tensorboard_target_dir = resolve_target_paths(
            run_dir,
            target_name,
            tensorboard_dir,
            args.suffix_conflicts,
        )
    except FileExistsError as exc:
        print(f"[skip] {run_dir.name}: {exc}")
        return False

    if args.dry_run:
        print(f"[dry-run] {run_dir.name} -> {target_dir.name}")
        if tensorboard_dir is not None and tensorboard_target_dir is not None:
            print(
                f"[dry-run] tensorboard {tensorboard_dir} "
                f"-> {tensorboard_target_dir}"
            )
        return True

    question = f"Rename {run_dir.name} -> {target_dir.name}?"
    if tensorboard_dir is not None and tensorboard_target_dir is not None:
        question = (
            f"Rename {run_dir.name} -> {target_dir.name} and TensorBoard "
            f"{tensorboard_dir.name} -> {tensorboard_target_dir.name}?"
        )
    if not args.yes and not confirm(question):
        print(f"[skip] {run_dir.name}: not confirmed")
        return False

    run_dir.rename(target_dir)
    if tensorboard_dir is not None and tensorboard_target_dir is not None:
        tensorboard_dir.rename(tensorboard_target_dir)
        print(f"[rename] tensorboard {tensorboard_dir} -> {tensorboard_target_dir}")
    print(f"[rename] {run_dir.name} -> {target_dir.name}")
    return True


def infer_run_timestamp(run_dir: Path) -> str:
    timestamp, _ = split_timestamp_prefix(run_dir.name)
    if timestamp is not None:
        return timestamp

    timestamp = time.strftime("%m%d%H%M%S", time.localtime(run_dir.stat().st_mtime))
    print(f"[infer] {run_dir.name}: timestamp -> {timestamp} from directory mtime")
    return timestamp


def infer_run_name(run_dir: Path, config: dict[str, Any]) -> str | None:
    logged_run_name = find_logged_run_name(run_dir)
    if logged_run_name is not RUN_NAME_NOT_FOUND:
        return logged_run_name

    run_name = infer_run_name_from_dir_name(run_dir.name, config)
    if run_name:
        print(f"[infer] {run_dir.name}: run_name -> {run_name}")
    return run_name


def infer_run_name_from_dir_name(
    dir_name: str,
    config: dict[str, Any],
) -> str | None:
    _, rest = split_timestamp_prefix(dir_name)
    if not rest:
        return None

    core_name = build_lora_run_core_name(config)
    if rest == core_name:
        return None
    if rest.startswith(f"{core_name}_"):
        return rest[len(core_name) + 1 :] or None

    if rest.lower() == "lora":
        return None
    if rest.lower().startswith("lora_"):
        return rest[5:] or None

    return None


def split_timestamp_prefix(dir_name: str) -> tuple[str | None, str]:
    full_timestamp = re.match(
        r"^(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?:[_-]|$)",
        dir_name,
    )
    if full_timestamp:
        timestamp = "".join(full_timestamp.groups()[1:])
        return timestamp, dir_name[full_timestamp.end() :].lstrip("_-")

    short_timestamp = re.match(
        r"^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:[_-]|$)",
        dir_name,
    )
    if short_timestamp:
        timestamp = "".join(short_timestamp.groups())
        return timestamp, dir_name[short_timestamp.end() :].lstrip("_-")

    return None, dir_name


RUN_NAME_NOT_FOUND = object()


def find_logged_run_name(run_dir: Path) -> str | None | object:
    log_file = run_dir / "log.txt"
    if not log_file.is_file():
        return RUN_NAME_NOT_FOUND

    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return RUN_NAME_NOT_FOUND

    for line in text.splitlines():
        match = re.search(r"\brun_name:\s*(.*?)\s*$", line)
        if not match:
            continue
        value = clean_log_value(match.group(1))
        if not value or value.lower() in {"none", "null"}:
            return None
        return value
    return RUN_NAME_NOT_FOUND


def find_tensorboard_dir(
    run_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    if args.no_tensorboard:
        return None

    candidates: list[Path] = []
    if args.tensorboard_root:
        candidates.append(Path(args.tensorboard_root).expanduser() / run_dir.name)

    logged_tensorboard_dir = find_logged_tensorboard_dir(run_dir)
    if logged_tensorboard_dir is not None:
        candidates.append(logged_tensorboard_dir)

    config_tensorboard_root = resolve_config_tensorboard_root(config, run_dir)
    if config_tensorboard_root is not None:
        candidates.append(config_tensorboard_root / run_dir.name)

    for candidate in unique_paths(candidates):
        if candidate.is_dir():
            return candidate

    if candidates:
        print(f"[skip-tb] {run_dir.name}: TensorBoard directory not found")
    return None


def resolve_target_paths(
    run_dir: Path,
    target_name: str,
    tensorboard_dir: Path | None,
    suffix_conflicts: bool,
) -> tuple[Path, Path | None]:
    suffix_index: int | None = None
    while True:
        candidate_name = (
            target_name if suffix_index is None else f"{target_name}__dup{suffix_index}"
        )
        target_dir = run_dir.with_name(candidate_name)
        tensorboard_target_dir = (
            tensorboard_dir.with_name(candidate_name)
            if tensorboard_dir is not None
            else None
        )

        conflicts = []
        if target_dir.exists():
            conflicts.append(f"log target exists ({candidate_name})")
        if tensorboard_target_dir is not None and tensorboard_target_dir.exists():
            conflicts.append(f"TensorBoard target exists ({tensorboard_target_dir})")

        if not conflicts:
            return target_dir, tensorboard_target_dir

        if not suffix_conflicts:
            raise FileExistsError(
                "; ".join(conflicts) + "; rerun with --suffix-conflicts to keep both"
            )
        suffix_index = 2 if suffix_index is None else suffix_index + 1


def find_logged_tensorboard_dir(run_dir: Path) -> Path | None:
    log_file = run_dir / "log.txt"
    if not log_file.is_file():
        return None

    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        match = re.search(r"\btensorboard_logdir:\s*(.+?)\s*$", line)
        if not match:
            continue
        value = clean_log_value(match.group(1))
        if not value or value.lower() == "none":
            continue
        path = Path(value).expanduser()
        if path.name == run_dir.name:
            return path
    return None


def resolve_config_tensorboard_root(
    config: dict[str, Any],
    run_dir: Path,
) -> Path | None:
    global_cfg = config.get("global")
    if not isinstance(global_cfg, Mapping):
        return None

    raw_root = global_cfg.get("tensorboard_logdir")
    if raw_root in (None, ""):
        return None

    root = Path(str(raw_root)).expanduser()
    if root.is_absolute():
        return root

    config_path = find_logged_config_path(run_dir)
    base_dir = config_path.parent if config_path is not None else TRAIN_DIR
    return (base_dir / root).resolve()


def find_logged_config_path(run_dir: Path) -> Path | None:
    log_file = run_dir / "log.txt"
    if not log_file.is_file():
        return None

    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        match = re.search(r"\bconfig_path:\s*(.+?)\s*$", line)
        if not match:
            continue
        value = clean_log_value(match.group(1))
        if value and value.lower() != "none":
            return Path(value).expanduser()
    return None


def clean_log_value(value: str) -> str:
    return value.strip().strip("'\"")


def unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.expanduser())
    return unique


def is_lora_run(run_dir: Path) -> bool:
    if "lora" in run_dir.name.lower():
        return True
    if MODEL_SIZE_PATTERN.search(run_dir.name) and re.search(
        r"(?:^|_)R-\d+",
        run_dir.name,
    ):
        return True

    log_file = run_dir / "log.txt"
    if not log_file.is_file():
        return False
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "active_method: lora" in text.lower()


def load_config(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return None

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("This script requires pyyaml to read config.yaml.") from exc

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def patch_missing_from_dir_name(config: dict[str, Any], dir_name: str) -> None:
    global_cfg = ensure_mapping(config, "global")
    lora_cfg = ensure_mapping(config, "LoRA")

    if not global_cfg.get("model_name"):
        match = MODEL_SIZE_PATTERN.search(dir_name)
        if match:
            global_cfg["model_name"] = match.group(1).replace("b", "B")

    if not lora_cfg.get("rank"):
        rank = find_first_match(
            dir_name,
            (
                r"(?:^|[_-])R[-_=]?(\d+)(?:$|[_-])",
                r"(?:^|[_-])(?:lora_)?r(?:ank)?[=-](\d+)(?:$|[_-])",
            ),
        )
        if rank:
            lora_cfg["rank"] = int(rank)

    if not lora_cfg.get("scaling_type"):
        scaling = find_first_match(dir_name, (r"scaling[=-](r[/_-]a|r[/_-]sqrta)",))
        if scaling:
            lora_cfg["scaling_type"] = scaling.replace("_", "/").replace("-", "/")

    if not lora_cfg.get("dropout_rate"):
        dropout = find_first_match(
            dir_name,
            (r"dropout(?:_rate)?[=-]([0-9.eE+-]+)(?:$|[_-])",),
        )
        if dropout:
            lora_cfg["dropout_rate"] = float(dropout)

    if not lora_cfg.get("target_modules"):
        module_code = find_first_match(dir_name.lower(), (r"(?:^|_)([qkvo]{1,4})(?:$|_)",))
        if module_code:
            lora_cfg["target_modules"] = [
                MODULE_CODE_TO_TARGET[letter]
                for letter in "qkvo"
                if letter in module_code
            ]


def ensure_layer_selection(
    config: dict[str, Any],
    dir_name: str,
    args: argparse.Namespace,
) -> None:
    lora_cfg = ensure_mapping(config, "LoRA")
    side = normalize_side(lora_cfg.get("target_layer_side"))
    count = lora_cfg.get("target_layer_count")
    count_value = parse_optional_int(count)

    if side == "all":
        lora_cfg["target_layer_side"] = "all"
        lora_cfg["target_layer_count"] = 0
        return

    if side in {"input", "output"} and count_value is not None:
        lora_cfg["target_layer_side"] = side
        lora_cfg["target_layer_count"] = count_value
        return

    if side is not None and side not in {"input", "output"}:
        raise ValueError("LoRA.target_layer_side must be one of: all, input, output.")

    inferred = infer_layer_selection_from_name(dir_name)
    if inferred is not None:
        inferred_side, inferred_count, label = inferred
        lora_cfg["target_layer_side"] = inferred_side
        lora_cfg["target_layer_count"] = inferred_count
        print(f"[infer] {dir_name}: layer selection -> {label}")
        return

    if side in {"input", "output"}:
        if args.infer_missing_layers or not sys.stdin.isatty():
            raise ValueError(
                "target_layer_side is set but target_layer_count is missing; "
                "rerun interactively or add top-N/end-N to the directory name"
            )
        print(
            f"[missing] {dir_name}: target_layer_side={side}, "
            "but target_layer_count is missing"
        )

    if side is None and count_value not in (None, 0):
        if args.infer_missing_layers or not sys.stdin.isatty():
            raise ValueError(
                "target_layer_count is set but target_layer_side is missing; "
                "rerun interactively or add top/end/full to the directory name"
            )
        print(
            f"[missing] {dir_name}: target_layer_count={count_value}, "
            "but top/end/full is missing"
        )

    if args.infer_missing_layers or not sys.stdin.isatty():
        lora_cfg["target_layer_side"] = "all"
        lora_cfg["target_layer_count"] = 0
        print(f"[infer] {dir_name}: missing layer selection, using full")
        return

    while True:
        answer = input(
            f"[missing] {dir_name}: enter layer selection "
            "(full, top-N, end-N; default full): "
        ).strip()
        if not answer:
            answer = "full"
        try:
            selected_side, selected_count, _ = parse_layer_selection(answer)
        except ValueError as exc:
            print(f"[error] {exc}")
            continue
        lora_cfg["target_layer_side"] = selected_side
        lora_cfg["target_layer_count"] = selected_count
        return


def infer_layer_selection_from_name(dir_name: str) -> tuple[str, int, str] | None:
    lowered = dir_name.lower()
    if re.search(r"(?:^|[_-])full(?:$|[_-])", lowered):
        return "all", 0, "full"

    patterns = (
        r"(?:^|[_-])(top)[-_]?(\d+)(?:$|[_-])",
        r"(?:^|[_-])(end)[-_]?(\d+)(?:$|[_-])",
        r"layers[=_-](input|front|first|top)[-_]?(\d+)(?:$|[_-])",
        r"layers[=_-](output|back|last|end)[-_]?(\d+)(?:$|[_-])",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        return parse_layer_selection(f"{match.group(1)}-{match.group(2)}")

    return None


def parse_layer_selection(value: str) -> tuple[str, int, str]:
    lowered = value.strip().lower()
    if lowered in {"full", "all", "none"}:
        return "all", 0, "full"

    match = re.fullmatch(r"(top|input|front|first)[-_]?(\d+)", lowered)
    if match:
        count = int(match.group(2))
        if count <= 0:
            raise ValueError("top-N must use N > 0")
        return "input", count, f"top-{count}"

    match = re.fullmatch(r"(end|output|back|last)[-_]?(\d+)", lowered)
    if match:
        count = int(match.group(2))
        if count <= 0:
            raise ValueError("end-N must use N > 0")
        return "output", count, f"end-{count}"

    raise ValueError("layer selection must be full, top-N, or end-N")


def normalize_side(value: Any) -> str | None:
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"all", "none", "full"}:
        return "all"
    if lowered in {"input", "front", "first", "top"}:
        return "input"
    if lowered in {"output", "back", "last", "end"}:
        return "output"
    return lowered


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        config[key] = {}
        return config[key]
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def find_first_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print("[skip] non-interactive mode requires --yes")
        return False
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
