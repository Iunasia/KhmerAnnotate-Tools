"""Fine-tune XLM-RoBERTa (or mDeBERTa-v3) for Khmer NER on CoNLL-2002 data.

Supports pre-split datasets (``--train`` + ``--dev`` [+ optional ``--test``])
or a single annotated file (``--input``) that is automatically split 80/10/10
grouped by document ID to guarantee zero article leakage.

Examples (run from the repository root):

    # Validate splits and check for leakage locally (no PyTorch required)
    python baseline/train_xlmr.py --train data/curated/train.conll --dev data/curated/dev.conll --dry-run

    # Fine-tune xlm-roberta-base on GPU (e.g. Google Colab T4)
    python baseline/train_xlmr.py \\
        --train data/curated/train.conll \\
        --dev data/curated/dev.conll \\
        --test data/curated/test.conll \\
        --output-dir outputs/xlmr_baseline

    # One-flag swap to mDeBERTa-v3 for comparison
    python baseline/train_xlmr.py \\
        --model-name microsoft/mdeberta-v3-base \\
        --train data/curated/train.conll --dev data/curated/dev.conll
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import (
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    LABELS,
    Sentence,
    dataset_summary,
    decode_word_predictions,
    encode_sentences,
    load_dataset_file,
    repair_bio_tags,
    split_by_document,
    validate_sentences,
    verify_no_leakage,
    write_conll,
)
from baseline.metrics import (
    analyze_errors,
    compute_ner_metrics,
    format_metrics_table,
    write_error_report_html,
)


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when installed."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_splits(
    args: argparse.Namespace,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    if args.input is not None:
        all_sentences = load_dataset_file(args.input, require_tags=True)
        train_sents, dev_sents, test_sents = split_by_document(
            all_sentences,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        return train_sents, dev_sents, test_sents

    if args.train is None or args.dev is None:
        raise SystemExit("Provide either --input <file> or both --train and --dev.")

    train_sents = load_dataset_file(args.train, require_tags=True)
    dev_sents = load_dataset_file(args.dev, require_tags=True)
    test_sents = load_dataset_file(args.test, require_tags=True) if args.test else []
    return train_sents, dev_sents, test_sents


def _check_splits(
    train_sents: Sequence[Sentence],
    dev_sents: Sequence[Sentence],
    test_sents: Sequence[Sentence],
    *,
    allow_bio_errors: bool = False,
    allow_leakage: bool = False,
) -> dict[str, Any]:
    """Validate BIO integrity and zero document leakage across splits."""
    validation: dict[str, dict[str, int]] = {}
    all_errors: list[str] = []

    for split_name, sentences in (("train", train_sents), ("dev", dev_sents), ("test", test_sents)):
        if not sentences:
            continue
        issues = validate_sentences(sentences)
        errors = [item for item in issues if item.severity == "error"]
        warnings = [item for item in issues if item.severity == "warning"]
        validation[split_name] = {"errors": len(errors), "warnings": len(warnings)}
        if warnings:
            print(
                f"  ! {split_name}: {len(warnings)} warning(s) "
                f"(e.g. sentence #{warnings[0].sentence_index + 1}: {warnings[0].message})"
            )
        if errors:
            for item in errors[:5]:
                all_errors.append(
                    f"{split_name} sent #{item.sentence_index + 1} tok {item.token_index}: "
                    f"{item.kind} ({item.message})"
                )

    if all_errors and not allow_bio_errors:
        preview = "\n  ".join(all_errors)
        raise SystemExit(
            f"BIO validation failed ({len(all_errors)} error(s) shown):\n  {preview}\n"
            "Fix the annotations or re-run with --allow-bio-errors."
        )

    leakage = verify_no_leakage(train_sents, dev_sents, test_sents)
    if leakage["has_doc_leakage"] and not allow_leakage:
        raise SystemExit(
            f"Document leakage detected across splits: {leakage['doc_overlap']}\n"
            "Re-split by document ID or pass --allow-leakage to override."
        )
    if leakage["has_text_leakage"]:
        print(f"  ! Warning: duplicate sentence text across splits: {leakage['text_overlap']}")

    return {"validation": validation, "leakage": leakage}


def _build_training_args(
    *,
    TrainingArguments: Any,
    output_dir: Path,
    args: argparse.Namespace,
    use_fp16: bool,
) -> Any:
    """Construct HF TrainingArguments compatible across transformers 4.40+ and 5.x."""
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.grad_accum,
        "max_grad_norm": args.max_grad_norm,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_strategy": "epoch",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": use_fp16,
        "report_to": "none",
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    # Drop any argument unsupported by an older/newer transformers version.
    supported_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**supported_kwargs)


def _extract_epoch_history(log_history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate HF Trainer log_history entries into one summary row per epoch."""
    by_epoch: dict[float, dict[str, Any]] = {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        rounded = round(float(epoch), 2)
        row = by_epoch.setdefault(rounded, {"epoch": rounded})
        for key in (
            "loss",
            "eval_loss",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "eval_per_f1",
            "eval_loc_f1",
            "eval_org_f1",
        ):
            if key in entry:
                row[key] = round(float(entry[key]), 6)
    return [by_epoch[ep] for ep in sorted(by_epoch)]


def predict_split(
    trainer: Any,
    encoded: Sequence[dict[str, Any]],
    sentences: Sequence[Sentence],
    *,
    repair_bio: bool = False,
) -> tuple[list[list[str]], int]:
    """Run Trainer.predict and map subword predictions back to word-level BIO tags."""
    import numpy as np

    class _InferenceDataset:
        def __init__(self, items: Sequence[dict[str, Any]]) -> None:
            self.items = [
                {k: v for k, v in item.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                for item in items
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self.items[idx]

    output = trainer.predict(_InferenceDataset(encoded))
    pred_ids_batch = np.argmax(output.predictions, axis=-1)

    word_predictions: list[list[str]] = []
    total_truncated_words = 0
    for pred_ids, item, sentence in zip(pred_ids_batch, encoded, sentences):
        tags, truncated = decode_word_predictions(
            pred_ids.tolist(),
            item["word_ids"],
            num_words=len(sentence.tokens),
            id2label=ID2LABEL,
        )
        if repair_bio:
            tags = repair_bio_tags(tags)
        word_predictions.append(tags)
        total_truncated_words += truncated
    return word_predictions, total_truncated_words


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa (or mDeBERTa-v3) for Khmer NER on CoNLL-2002 data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, help="single annotated CoNLL/JSONL file to auto-split by doc_id")
    parser.add_argument("--train", type=Path, help="training split (.conll or .jsonl)")
    parser.add_argument("--dev", type=Path, help="development/validation split (.conll or .jsonl)")
    parser.add_argument("--test", type=Path, help="optional test split (.conll or .jsonl)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument(
        "--model-name",
        default="xlm-roberta-base",
        help="Hugging Face checkpoint (default: xlm-roberta-base; try microsoft/mdeberta-v3-base)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xlmr_baseline"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-all-tokens", action="store_true")
    parser.add_argument("--no-fp16", action="store_true", help="disable mixed precision even on CUDA")
    parser.add_argument("--repair-bio", action="store_true", help="convert orphan I-X predictions to B-X")
    parser.add_argument("--allow-bio-errors", action="store_true")
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data, check leakage, and print summary without loading PyTorch/transformers",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_global_seed(args.seed)

    train_sents, dev_sents, test_sents = _load_splits(args)
    if not train_sents or not dev_sents:
        raise SystemExit("Both train and dev splits must contain at least 1 sentence.")

    checks = _check_splits(
        train_sents,
        dev_sents,
        test_sents,
        allow_bio_errors=args.allow_bio_errors,
        allow_leakage=args.allow_leakage,
    )

    summaries = {
        "train": dataset_summary(train_sents),
        "dev": dataset_summary(dev_sents),
        "test": dataset_summary(test_sents) if test_sents else None,
    }
    print("Dataset summary:")
    for name in ("train", "dev", "test"):
        info = summaries[name]
        if info:
            print(
                f"  {name:<5}: {info['sentences']} sents, {info['tokens']} tokens, "
                f"{info['documents']} docs, entities={info['entity_counts']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input is not None:
        splits_dir = args.output_dir / "splits"
        write_conll(train_sents, splits_dir / "train.conll")
        write_conll(dev_sents, splits_dir / "dev.conll")
        if test_sents:
            write_conll(test_sents, splits_dir / "test.conll")
        print(f"Saved leak-free document splits to {splits_dir}/")

    if args.dry_run:
        print("Dry run complete: data valid and leakage checks passed.")
        return 0

    try:
        import numpy as np
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            f"Missing ML dependency ({exc}). Install with:\n"
            "  pip install -r baseline/requirements-baseline.txt"
        ) from exc

    device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    print(f"Loading {args.model_name} on {device_label} (fp16={use_fp16})...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_encoded, train_trunc = encode_sentences(
        train_sents,
        tokenizer,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
    )
    dev_encoded, dev_trunc = encode_sentences(
        dev_sents,
        tokenizer,
        max_length=args.max_length,
        label_all_tokens=False,
    )
    test_encoded, test_trunc = (
        encode_sentences(
            test_sents,
            tokenizer,
            max_length=args.max_length,
            label_all_tokens=False,
        )
        if test_sents
        else ([], {"sentences": 0, "truncated_sentences": 0, "truncated_words": 0})
    )

    for split_name, trunc in (("train", train_trunc), ("dev", dev_trunc), ("test", test_trunc)):
        if trunc["truncated_sentences"]:
            print(
                f"  ! {split_name}: {trunc['truncated_sentences']} sentence(s) truncated "
                f"at max_length={args.max_length} ({trunc['truncated_words']} words dropped)"
            )

    class _TokenDataset(torch.utils.data.Dataset):
        def __init__(self, items: Sequence[dict[str, Any]]) -> None:
            self.items = [
                {k: v for k, v in item.items() if k != "word_ids"}
                for item in items
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self.items[idx]

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_sequences: list[list[str]] = []
        pred_sequences: list[list[str]] = []
        for pred_row, label_row in zip(predictions, labels):
            sent_true: list[str] = []
            sent_pred: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if int(label_id) == IGNORE_INDEX:
                    continue
                sent_true.append(ID2LABEL[int(label_id)])
                sent_pred.append(ID2LABEL[int(pred_id)])
            true_sequences.append(sent_true)
            pred_sequences.append(sent_pred)

        metrics = compute_ner_metrics(true_sequences, pred_sequences)
        overall = metrics["overall"]
        result = {
            "precision": float(overall["precision"]),
            "recall": float(overall["recall"]),
            "f1": float(overall["f1"]),
            "macro_f1": float(overall["macro_f1"]),
        }
        for etype, stats in metrics["per_entity"].items():
            result[f"{etype.lower()}_f1"] = float(stats["f1"])
        return result

    training_args = _build_training_args(
        TrainingArguments=TrainingArguments,
        output_dir=args.output_dir,
        args=args,
        use_fp16=use_fp16,
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": _TokenDataset(train_encoded),
        "eval_dataset": _TokenDataset(dev_encoded),
        "data_collator": DataCollatorForTokenClassification(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    best_dir = args.output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\nSaved best checkpoint to {best_dir}")

    history = _extract_epoch_history(trainer.state.log_history)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dev_preds, dev_pred_trunc = predict_split(
        trainer, dev_encoded, dev_sents, repair_bio=args.repair_bio
    )
    dev_metrics = compute_ner_metrics([s.tags for s in dev_sents], dev_preds)
    dev_errors = analyze_errors(dev_sents, dev_preds)
    write_conll(dev_sents, args.output_dir / "dev_predictions.conll", predicted_tags=dev_preds)
    write_error_report_html(
        dev_metrics,
        dev_errors,
        args.output_dir / "dev_error_report.html",
        title=f"KhmerAnnotate — {args.model_name} Dev Error Report",
    )
    print("\nDev evaluation (word-level exact spans):")
    print(format_metrics_table(dev_metrics))

    test_metrics = None
    test_pred_trunc = 0
    if test_sents:
        test_preds, test_pred_trunc = predict_split(
            trainer, test_encoded, test_sents, repair_bio=args.repair_bio
        )
        test_metrics = compute_ner_metrics([s.tags for s in test_sents], test_preds)
        test_errors = analyze_errors(test_sents, test_preds)
        write_conll(test_sents, args.output_dir / "test_predictions.conll", predicted_tags=test_preds)
        (args.output_dir / "test_metrics.json").write_text(
            json.dumps(
                {"metrics": test_metrics, "error_counts": test_errors["counts"]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_error_report_html(
            test_metrics,
            test_errors,
            args.output_dir / "test_error_report.html",
            title=f"KhmerAnnotate — {args.model_name} Test Error Report",
        )
        print("\nTest evaluation (word-level exact spans):")
        print(format_metrics_table(test_metrics))

    summary_payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": device_label,
        "fp16": use_fp16,
        "dataset_summary": summaries,
        "checks": checks,
        "truncation": {
            "train": train_trunc,
            "dev": dev_trunc,
            "test": test_trunc,
            "dev_eval_dropped_words": dev_pred_trunc,
            "test_eval_dropped_words": test_pred_trunc,
        },
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote run summary to {args.output_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
