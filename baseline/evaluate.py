"""Evaluate a trained NER checkpoint or a saved predictions.conll file.

Exports:
    - ``metrics.json`` (overall & per-entity Precision, Recall, F1, confusion matrix, error counts)
    - ``error_report.html`` (UTF-8 Khmer side-by-side Gold vs Predicted span report)
    - ``predictions.conll`` (when evaluating from a model checkpoint)
    - Up to 4 high-resolution (300 DPI) PNG charts when ``matplotlib`` is installed:
        1. ``per_entity_prf1.png``
        2. ``entity_confusion_matrix.png``
        3. ``error_breakdown.png``
        4. ``training_curves.png`` (when ``--history`` is supplied)

Examples (run from the repository root):

    # Evaluate a trained checkpoint on test.conll
    python baseline/evaluate.py \\
        --model-dir outputs/xlmr_baseline/best_model \\
        --data data/curated/test.conll \\
        --history outputs/xlmr_baseline/training_history.json \\
        --output-dir outputs/xlmr_baseline/eval

    # Evaluate locally from saved 3-column predictions.conll (no PyTorch needed)
    python baseline/evaluate.py \\
        --predictions outputs/xlmr_baseline/test_predictions.conll \\
        --history outputs/xlmr_baseline/training_history.json \\
        --output-dir outputs/xlmr_baseline/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import (
    ID2LABEL,
    Sentence,
    decode_word_predictions,
    load_dataset_file,
    repair_bio_tags,
    write_conll,
)
from baseline.metrics import (
    analyze_errors,
    compute_ner_metrics,
    format_metrics_table,
    write_error_report_html,
)


def read_predictions_conll(path: Path) -> tuple[list[Sentence], list[list[str]]]:
    """Read a 3-column CoNLL file (``token<TAB>gold<TAB>pred``)."""
    sentences: list[Sentence] = []
    predictions: list[list[str]] = []

    tokens: list[str] = []
    gold_tags: list[str] = []
    pred_tags: list[str] = []
    meta: dict[str, str] = {}

    def flush() -> None:
        nonlocal tokens, gold_tags, pred_tags, meta
        if not tokens:
            return
        sentences.append(
            Sentence(
                tokens=tokens,
                tags=gold_tags,
                doc_id=meta.get("doc_id", ""),
                sent_id=meta.get("sent_id", ""),
                meta=dict(meta),
            )
        )
        predictions.append(pred_tags)
        tokens = []
        gold_tags = []
        pred_tags = []
        meta = {}

    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line.startswith("#"):
                if "=" in line:
                    key, _, value = line[1:].partition("=")
                    meta[key.strip().lower()] = value.strip()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if parts[0] == "-DOCSTART-":
                flush()
                continue
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 3 columns (token, gold, pred), got {line!r}"
                )
            tokens.append(parts[0])
            gold_tags.append(parts[-2])
            pred_tags.append(parts[-1])

    flush()
    return sentences, predictions


def run_model_inference(
    model_dir: Path,
    sentences: Sequence[Sentence],
    *,
    max_length: int = 256,
    batch_size: int = 32,
    repair_bio: bool = False,
) -> tuple[list[list[str]], int]:
    """Run batched token-classification inference and decode to word-level BIO tags."""
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"Missing ML dependency ({exc}). Install with:\n"
            "  pip install -r baseline/requirements-baseline.txt"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id2label = {
        int(k): v for k, v in getattr(model.config, "id2label", ID2LABEL).items()
    }

    all_preds: list[list[str]] = []
    truncated_words = 0

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        encoded = tokenizer(
            [s.tokens for s in chunk],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.no_grad():
            logits = model(**model_inputs).logits
        pred_ids_batch = logits.argmax(dim=-1).cpu().tolist()

        for index, (pred_ids, sentence) in enumerate(zip(pred_ids_batch, chunk)):
            word_ids = encoded.word_ids(batch_index=index)
            tags, dropped = decode_word_predictions(
                pred_ids,
                word_ids,
                num_words=len(sentence.tokens),
                id2label=id2label,
            )
            if repair_bio:
                tags = repair_bio_tags(tags)
            all_preds.append(tags)
            truncated_words += dropped

    return all_preds, truncated_words


def generate_charts(
    metrics: dict[str, Any],
    error_analysis: dict[str, Any],
    output_dir: Path,
    *,
    history: Sequence[dict[str, Any]] | None = None,
    target_f1: float = 0.70,
    dpi: int = 300,
) -> list[Path]:
    """Generate 300-DPI evaluation PNG charts when ``matplotlib`` is installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ! matplotlib not installed; skipping PNG charts.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Per-Entity Precision / Recall / F1 grouped bar chart
    categories = [*metrics["per_entity"].keys(), "Overall"]
    precisions = [metrics["per_entity"][k]["precision"] for k in metrics["per_entity"]] + [
        metrics["overall"]["precision"]
    ]
    recalls = [metrics["per_entity"][k]["recall"] for k in metrics["per_entity"]] + [
        metrics["overall"]["recall"]
    ]
    f1s = [metrics["per_entity"][k]["f1"] for k in metrics["per_entity"]] + [
        metrics["overall"]["f1"]
    ]

    x_positions = list(range(len(categories)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar([x - width for x in x_positions], precisions, width, label="Precision", color="#3b82f6")
    b2 = ax.bar(x_positions, recalls, width, label="Recall", color="#10b981")
    b3 = ax.bar([x + width for x in x_positions], f1s, width, label="F1", color="#f59e0b")
    ax.axhline(target_f1, linestyle="--", color="#ef4444", linewidth=1.2, label=f"Target F1 ({target_f1:.2f})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Khmer NER Performance by Entity Type")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bars in (b1, b2, b3):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    chart1 = output_dir / "per_entity_prf1.png"
    fig.tight_layout()
    fig.savefig(chart1, dpi=dpi)
    plt.close(fig)
    saved.append(chart1)

    # 2. Entity-level Confusion Matrix
    labels = error_analysis["confusion_labels"]
    matrix = [
        [error_analysis["confusion_matrix"][row][col] for col in labels]
        for row in labels
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Entity")
    ax.set_ylabel("Gold Entity")
    ax.set_title("Entity-Level Confusion Matrix (Span Alignment)")
    max_val = max((max(row) for row in matrix), default=1) or 1
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            color = "white" if val > max_val * 0.5 else "#111827"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    chart2 = output_dir / "entity_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(chart2, dpi=dpi)
    plt.close(fig)
    saved.append(chart2)

    # 3. Span Error Breakdown
    counts = error_analysis["counts"]
    err_labels = [
        ("Correct", counts["correct"], "#10b981"),
        ("Boundary Error", counts["boundary_error"], "#f59e0b"),
        ("Type Error", counts["type_error"], "#6366f1"),
        ("Boundary + Type", counts["boundary_and_type_error"], "#ec4899"),
        ("Missed (FN)", counts["missed"], "#ef4444"),
        ("Spurious (FP)", counts["spurious"], "#64748b"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_pos = list(range(len(err_labels)))
    values = [item[1] for item in err_labels]
    bars = ax.barh(y_pos, values, color=[item[2] for item in err_labels])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([item[0] for item in err_labels])
    ax.invert_yaxis()
    ax.set_xlabel("Entity Span Count")
    ax.set_title("NER Span-Level Error Breakdown")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, values):
        ax.annotate(
            str(val),
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    chart3 = output_dir / "error_breakdown.png"
    fig.tight_layout()
    fig.savefig(chart3, dpi=dpi)
    plt.close(fig)
    saved.append(chart3)

    # 4. Training Curves (optional, when training_history.json is provided)
    if history:
        epochs = [row["epoch"] for row in history if "epoch" in row]
        train_loss = [(row["epoch"], row["loss"]) for row in history if "loss" in row]
        eval_loss = [(row["epoch"], row["eval_loss"]) for row in history if "eval_loss" in row]
        eval_f1 = [(row["epoch"], row["eval_f1"]) for row in history if "eval_f1" in row]

        if epochs and (train_loss or eval_loss or eval_f1):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
            if train_loss:
                ax1.plot([p[0] for p in train_loss], [p[1] for p in train_loss], marker="o", label="Train Loss")
            if eval_loss:
                ax1.plot([p[0] for p in eval_loss], [p[1] for p in eval_loss], marker="s", label="Dev Loss")
            ax1.set_title("Loss per Epoch")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(linestyle=":", alpha=0.5)

            if eval_f1:
                ax2.plot([p[0] for p in eval_f1], [p[1] for p in eval_f1], marker="o", color="#10b981", label="Dev F1")
                ax2.axhline(target_f1, linestyle="--", color="#ef4444", linewidth=1.2, label=f"Target ({target_f1:.2f})")
            ax2.set_title("Dev Span F1 per Epoch")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("F1")
            ax2.set_ylim(0.0, 1.05)
            ax2.legend()
            ax2.grid(linestyle=":", alpha=0.5)

            chart4 = output_dir / "training_curves.png"
            fig.tight_layout()
            fig.savefig(chart4, dpi=dpi)
            plt.close(fig)
            saved.append(chart4)

    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Khmer NER checkpoint or saved predictions.conll.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, help="saved Hugging Face checkpoint directory")
    parser.add_argument("--data", type=Path, help="labeled CoNLL-2002 / JSONL file (required with --model-dir)")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="3-column CoNLL file (token, gold, pred) to evaluate without loading a model",
    )
    parser.add_argument("--history", type=Path, help="optional training_history.json for training curves chart")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xlmr_eval"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-f1", type=float, default=0.70)
    parser.add_argument("--repair-bio", action="store_true")
    parser.add_argument("--strict-iob2", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--max-html-examples", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    truncated_words = 0
    if args.predictions is not None:
        sentences, pred_tags = read_predictions_conll(args.predictions)
        if args.repair_bio:
            pred_tags = [repair_bio_tags(tags) for tags in pred_tags]
    elif args.model_dir is not None and args.data is not None:
        sentences = load_dataset_file(args.data, require_tags=True)
        pred_tags, truncated_words = run_model_inference(
            args.model_dir,
            sentences,
            max_length=args.max_length,
            batch_size=args.batch_size,
            repair_bio=args.repair_bio,
        )
    else:
        raise SystemExit("Provide either --predictions <file> OR both --model-dir and --data.")

    if not sentences:
        raise SystemExit("Evaluation dataset is empty.")

    metrics = compute_ner_metrics(
        [s.tags for s in sentences],
        pred_tags,
        strict_iob2=args.strict_iob2,
    )
    error_analysis = analyze_errors(
        sentences,
        pred_tags,
        strict_iob2=args.strict_iob2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predictions is None:
        write_conll(sentences, args.output_dir / "predictions.conll", predicted_tags=pred_tags)

    metrics_payload = {
        "sentences": len(sentences),
        "truncated_words": truncated_words,
        "metrics": metrics,
        "error_counts": error_analysis["counts"],
        "confusion_labels": error_analysis["confusion_labels"],
        "confusion_matrix": error_analysis["confusion_matrix"],
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    html_path = write_error_report_html(
        metrics,
        error_analysis,
        args.output_dir / "error_report.html",
        max_examples=args.max_html_examples,
    )

    history = None
    if args.history is not None and args.history.is_file():
        history = json.loads(args.history.read_text(encoding="utf-8"))

    charts: list[Path] = []
    if not args.no_charts:
        charts = generate_charts(
            metrics,
            error_analysis,
            args.output_dir / "charts",
            history=history,
            target_f1=args.target_f1,
        )

    print(format_metrics_table(metrics))
    if truncated_words:
        print(f"\nWarning: {truncated_words} trailing words truncated at max_length={args.max_length}.")
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved HTML error report to {html_path}")
    if charts:
        print(f"Saved {len(charts)} chart(s) in {args.output_dir / 'charts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
