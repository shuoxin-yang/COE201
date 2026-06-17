from __future__ import annotations

import json
import math
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "Report_assets"
EXTRACTED_DIR = ASSET_DIR / "extracted"
PDF_PATH = ROOT / "汇报.pdf"
RUN_DIR = (
    ROOT
    / "Train"
    / "logs"
    / "0531144908_2B_R-4_DR-0.5_full_qkvo_DropoutTest"
)

PDF_IMAGES = {
    "data_length_distribution.png": (14, 131),
    "semantic_similarity_filtering.jpeg": (15, 138),
    "filtering_pipeline.png": (17, 153),
    "model_comparison_small.jpeg": (20, 175),
    "model_comparison_large.jpeg": (20, 176),
    "overfitting_vs_parameters.png": (23, 206),
    "partial_finetuning_test.png": (24, 213),
    "partial_finetuning_train.png": (24, 214),
    "dropout_rank4.jpeg": (25, 220),
    "dropout_rank8.jpeg": (25, 223),
    "dropout_rank16.jpeg": (25, 224),
    "augmentation_only.png": (29, 251),
    "augmentation_dropout.png": (30, 256),
    "before_finetuning.png": (32, 265),
    "after_finetuning.png": (33, 271),
}


def extract_pdf_images() -> None:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with fitz.open(PDF_PATH) as document:
        for filename, (_, xref) in PDF_IMAGES.items():
            image = document.extract_image(xref)
            (EXTRACTED_DIR / filename).write_bytes(image["image"])


def labeled_panel(path: Path, label: str, width: int, height: int) -> Image.Image:
    source = Image.open(path).convert("RGB")
    scale = min((width - 30) / source.width, (height - 45) / source.height)
    source = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(source, ((width - source.width) // 2, 35))
    ImageDraw.Draw(panel).text((12, 10), label, fill="black")
    return panel


def make_grid(
    output_name: str,
    entries: list[tuple[str, str]],
    columns: int,
    panel_size: tuple[int, int],
) -> None:
    panels = [
        labeled_panel(EXTRACTED_DIR / filename, label, *panel_size)
        for filename, label in entries
    ]
    rows = math.ceil(len(panels) / columns)
    grid = Image.new(
        "RGB",
        (columns * panel_size[0], rows * panel_size[1]),
        (235, 235, 235),
    )
    for index, panel in enumerate(panels):
        grid.paste(
            panel,
            (
                (index % columns) * panel_size[0],
                (index // columns) * panel_size[1],
            ),
        )
    grid.save(ASSET_DIR / output_name, quality=94)


def make_qualitative_figures(width: int = 1800) -> None:
    for source_name, output_name in (
        ("before_finetuning.png", "qualitative_before_finetuning.jpg"),
        ("after_finetuning.png", "qualitative_after_finetuning.jpg"),
    ):
        source = Image.open(EXTRACTED_DIR / source_name).convert("RGB")
        scale = width / source.width
        output = source.resize(
            (width, round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        output.save(ASSET_DIR / output_name, quality=94)


def make_composite_figures() -> None:
    make_grid(
        "model_scale_comparison.jpg",
        [
            ("model_comparison_small.jpeg", "(a) Comparable trainable parameters: 0.5M-1.0M"),
            ("model_comparison_large.jpeg", "(b) Comparable trainable parameters: 1.0M-2.0M"),
        ],
        columns=2,
        panel_size=(980, 640),
    )
    make_grid(
        "partial_finetuning_comparison.jpg",
        [
            ("partial_finetuning_test.png", "(a) Test loss"),
            ("partial_finetuning_train.png", "(b) Training loss"),
        ],
        columns=2,
        panel_size=(900, 400),
    )
    make_grid(
        "dropout_rank_comparison.jpg",
        [
            ("dropout_rank4.jpeg", "(a) LoRA rank 4"),
            ("dropout_rank8.jpeg", "(b) LoRA rank 8"),
            ("dropout_rank16.jpeg", "(c) LoRA rank 16"),
        ],
        columns=3,
        panel_size=(580, 480),
    )
    make_grid(
        "augmentation_comparison.jpg",
        [
            ("augmentation_only.png", "(a) Original vs. augmented data"),
            ("augmentation_dropout.png", "(b) Augmentation and dropout"),
        ],
        columns=2,
        panel_size=(800, 600),
    )
    make_qualitative_figures()


def read_test_losses() -> list[tuple[int, float]]:
    rows = []
    checkpoint_dir = RUN_DIR / "checkpoints"
    for path in checkpoint_dir.glob("test_epoch_*_results.json"):
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append((int(metrics["epoch"]), float(metrics["test_loss"])))
    return sorted(rows)


def plot_final_training_curve() -> None:
    rows = read_test_losses()
    epochs = [epoch for epoch, _ in rows]
    losses = [loss for _, loss in rows]
    perplexities = [math.exp(loss) for loss in losses]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(epochs, losses, marker="o", linewidth=2, color="#126782")
    axes[0].set_title("Fixed-Test Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, perplexities, marker="o", linewidth=2, color="#d1495b")
    axes[1].set_title("Fixed-Test Perplexity by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Final Qwen3.5-2B LoRA Run (r=4, alpha=8, dropout=0.5)")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "final_training_metrics.png", dpi=180)
    plt.close(fig)


def plot_repository_data_counts() -> None:
    counts = {
        "Raw QA": 9615,
        "Similarity-filtered": 6227,
        "Score >= 3": 5913,
        "Augmented training corpus": 48075,
    }
    labels = list(counts)
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, values, color=["#5b8e7d", "#8cb369", "#f4a259", "#bc4b51"])
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    ax.set_ylabel("Number of QA pairs")
    ax.set_title("Repository Dataset Scale at Major Processing Stages")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "repository_dataset_scale.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    extract_pdf_images()
    make_composite_figures()
    plot_final_training_curve()
    plot_repository_data_counts()
    print(f"Report figures generated in {ASSET_DIR}")


if __name__ == "__main__":
    main()
