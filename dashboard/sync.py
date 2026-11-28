"""Push checker/ and baseline/ JSON artifacts directly into Neon PostgreSQL (or local SQLite).

Works both as a CLI (from local terminal or Google Colab) and as reusable
ingestion functions imported by ``dashboard/app.py``.

Examples (set DATABASE_URL to your Neon connection string, or omit for local SQLite):

    # Print current Desired-vs-Actual readiness scorecard
    python dashboard/sync.py scorecard

    # Push Stage 4 IAA report
    python dashboard/sync.py iaa --file outputs/qa/iaa/iaa_metrics.json --batch iaa_double_200 --by Sousdey

    # Push Stage 4 Label Checker QA report
    python dashboard/sync.py qa --file outputs/qa/labels/qa_report.json --batch curated_v1 --by Sousdey

    # Push Stage 4 leak-free split report
    python dashboard/sync.py split --file data/curated/split_report.json --name v1-curated --sources "VOA Khmer" --licence "Public Domain" --by Sousdey

    # Push Stage 5 XLM-R training + test evaluation metrics straight from Colab
    python dashboard/sync.py train \\
        --summary outputs/xlmr_baseline/run_summary.json \\
        --eval outputs/xlmr_baseline/eval/metrics.json \\
        --run-name xlmr-v1-seed42 --snapshot v1-curated --by Sousdey
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.db import QualityTrackerDB


def ingest_iaa_payload(
    db: QualityTrackerDB,
    payload: dict[str, Any],
    *,
    batch_name: str,
    recorded_by: str = "team",
    notes: str = "",
) -> int:
    """Ingest JSON output from ``checker/agreement.py`` into ``qa_checks``."""
    pairs = payload.get("pairs", [])
    annotator_desc = (
        ", ".join(f"{p.get('annotator_a', '?')} vs {p.get('annotator_b', '?')}" for p in pairs)
        if pairs
        else ""
    )
    sentences = int(pairs[0].get("sentences", 0)) if pairs else int(payload.get("sentences", 0))
    span_f1 = payload.get("mean_span_f1")
    token_kappa = payload.get("mean_token_kappa")
    entity_kappa = payload.get("mean_entity_only_kappa")
    target = float(payload.get("target_iaa", 0.80))
    passed = bool(payload.get("passed", (span_f1 is not None and float(span_f1) >= target)))

    return db.record_qa_check(
        check_type="iaa",
        batch_name=batch_name,
        annotators=annotator_desc,
        sentences_checked=sentences,
        span_f1=float(span_f1) if span_f1 is not None else None,
        token_kappa=float(token_kappa) if token_kappa is not None else None,
        entity_only_kappa=float(entity_kappa) if entity_kappa is not None else None,
        error_count=0,
        warning_count=sum(int(p.get("disagreement_sentences", 0)) for p in pairs),
        passed=passed,
        recorded_by=recorded_by,
        notes=notes,
    )


def ingest_qa_payload(
    db: QualityTrackerDB,
    payload: dict[str, Any],
    *,
    batch_name: str,
    recorded_by: str = "team",
    notes: str = "",
) -> int:
    """Ingest JSON output from ``checker/label_checker.py`` into ``qa_checks``."""
    error_count = int(payload.get("error_count", 0))
    warning_count = int(payload.get("warning_count", 0))
    return db.record_qa_check(
        check_type="label_audit",
        batch_name=batch_name,
        annotators="automated-label-checker",
        sentences_checked=int(payload.get("sentences", 0)),
        error_count=error_count,
        warning_count=warning_count,
        passed=bool(payload.get("passed", error_count == 0)),
        recorded_by=recorded_by,
        notes=notes,
    )


def ingest_split_payload(
    db: QualityTrackerDB,
    payload: dict[str, Any],
    *,
    snapshot_name: str,
    sources: str = "VOA Khmer",
    licence: str = "Public Domain",
    raw_articles: int = 0,
    cleaned_sentences: int = 0,
    segmented_sentences: int = 0,
    recorded_by: str = "team",
    notes: str = "",
) -> int:
    """Ingest JSON output from ``checker/prepare_splits.py`` into ``dataset_snapshots``."""
    splits = payload.get("splits", {})
    train = splits.get("train", {})
    dev = splits.get("dev", {})
    test = splits.get("test", {})

    train_s = int(train.get("sentences", 0))
    dev_s = int(dev.get("sentences", 0))
    test_s = int(test.get("sentences", 0))
    annotated_total = train_s + dev_s + test_s

    def _sum_entity(etype: str) -> int:
        return sum(
            int(split.get("entity_counts", {}).get(etype, 0))
            for split in (train, dev, test)
        )

    per_n = _sum_entity("PER")
    loc_n = _sum_entity("LOC")
    org_n = _sum_entity("ORG")
    qa_counts = payload.get("qa", {}).get("counts_by_kind", {})
    broken_clusters = int(qa_counts.get("broken-khmer-cluster", 0))
    has_leakage = bool(payload.get("leakage", {}).get("has_doc_leakage", False))

    return db.record_snapshot(
        snapshot_name=snapshot_name,
        sources=sources,
        licence=licence,
        raw_articles=raw_articles,
        cleaned_sentences=cleaned_sentences,
        segmented_sentences=segmented_sentences or annotated_total,
        annotated_sentences=annotated_total,
        train_sentences=train_s,
        dev_sentences=dev_s,
        test_sentences=test_s,
        total_entities=per_n + loc_n + org_n,
        per_entities=per_n,
        loc_entities=loc_n,
        org_entities=org_n,
        broken_khmer_clusters=broken_clusters,
        has_doc_leakage=has_leakage,
        seed=int(payload.get("seed", 42)),
        recorded_by=recorded_by,
        notes=notes,
    )


def ingest_training_payload(
    db: QualityTrackerDB,
    *,
    run_name: str,
    summary_payload: dict[str, Any] | None = None,
    eval_payload: dict[str, Any] | None = None,
    snapshot_name: str = "",
    split_evaluated: str | None = None,
    target_f1: float = 0.70,
    recorded_by: str = "team",
    notes: str = "",
) -> int:
    """Ingest ``run_summary.json`` and/or ``metrics.json`` from ``baseline/`` into ``training_runs``."""
    summary = summary_payload or {}
    eval_data = eval_payload or {}

    if "metrics" in eval_data:
        metrics = eval_data["metrics"]
        chosen_split = split_evaluated or "test"
    elif summary.get("test_metrics"):
        metrics = summary["test_metrics"]
        chosen_split = split_evaluated or "test"
    elif summary.get("dev_metrics"):
        metrics = summary["dev_metrics"]
        chosen_split = split_evaluated or "dev"
    else:
        raise ValueError("No metrics found in summary_payload or eval_payload.")

    overall = metrics.get("overall", {})
    per_entity = metrics.get("per_entity", {})
    err_counts = eval_data.get("error_counts", {})

    return db.record_training_run(
        run_name=run_name,
        snapshot_name=snapshot_name,
        model_name=str(summary.get("model_name", "xlm-roberta-base")),
        split_evaluated=chosen_split,
        seed=int(summary.get("seed", 42)),
        epochs=int(summary.get("epochs", 8)),
        batch_size=int(summary.get("batch_size", 16)),
        learning_rate=float(summary.get("learning_rate", 3e-5)),
        max_length=int(summary.get("max_length", 256)),
        device=str(summary.get("device", "GPU")),
        overall_precision=float(overall.get("precision", 0.0)),
        overall_recall=float(overall.get("recall", 0.0)),
        overall_f1=float(overall.get("f1", 0.0)),
        macro_f1=float(overall.get("macro_f1", 0.0)),
        per_f1=float(per_entity.get("PER", {}).get("f1", 0.0)),
        loc_f1=float(per_entity.get("LOC", {}).get("f1", 0.0)),
        org_f1=float(per_entity.get("ORG", {}).get("f1", 0.0)),
        err_correct=int(err_counts.get("correct", 0)),
        err_boundary=int(err_counts.get("boundary_error", 0)),
        err_type=int(err_counts.get("type_error", 0)),
        err_boundary_type=int(err_counts.get("boundary_and_type_error", 0)),
        err_missed=int(err_counts.get("missed", 0)),
        err_spurious=int(err_counts.get("spurious", 0)),
        target_f1=target_f1,
        recorded_by=recorded_by,
        notes=notes,
    )


def format_scorecard(scorecard: dict[str, Any]) -> str:
    """Format a compact ASCII table of Desired vs Actual quality targets."""
    lines = [
        f"Capstone Readiness: {scorecard['readiness_pct']:.1f}% "
        f"({scorecard['passed']} PASS / {scorecard['failed']} FAIL / {scorecard['pending']} PENDING)",
        f"{'Stage':<22} {'Target':<38} {'Expected':>10} {'Actual':>10} {'Status':>8}",
        "-" * 92,
    ]
    for item in scorecard["items"]:
        exp_str = f"{item['operator']} {item['expected_value']:g}"
        act_str = "—" if item["actual_value"] is None else f"{item['actual_value']:g}"
        lines.append(
            f"{item['stage']:<22} {item['label'][:38]:<38} {exp_str:>10} {act_str:>10} {item['status']:>8}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync checker/ and baseline/ JSON results to Neon PostgreSQL (or local SQLite).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Neon PostgreSQL URL or SQLite path (default: DATABASE_URL env var, else data/dashboard_local.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize tables and seed default quality targets")
    sub.add_parser("scorecard", help="Print the Desired vs Actual readiness scorecard")

    p_iaa = sub.add_parser("iaa", help="Ingest checker/agreement.py iaa_metrics.json")
    p_iaa.add_argument("--file", type=Path, required=True)
    p_iaa.add_argument("--batch", required=True)
    p_iaa.add_argument("--by", default="team")
    p_iaa.add_argument("--notes", default="")

    p_qa = sub.add_parser("qa", help="Ingest checker/label_checker.py qa_report.json")
    p_qa.add_argument("--file", type=Path, required=True)
    p_qa.add_argument("--batch", required=True)
    p_qa.add_argument("--by", default="team")
    p_qa.add_argument("--notes", default="")

    p_split = sub.add_parser("split", help="Ingest checker/prepare_splits.py split_report.json")
    p_split.add_argument("--file", type=Path, required=True)
    p_split.add_argument("--name", required=True)
    p_split.add_argument("--sources", default="VOA Khmer")
    p_split.add_argument("--licence", default="Public Domain")
    p_split.add_argument("--raw-articles", type=int, default=0)
    p_split.add_argument("--cleaned-sentences", type=int, default=0)
    p_split.add_argument("--segmented-sentences", type=int, default=0)
    p_split.add_argument("--by", default="team")
    p_split.add_argument("--notes", default="")

    p_train = sub.add_parser("train", help="Ingest baseline run_summary.json and/or eval metrics.json")
    p_train.add_argument("--summary", type=Path)
    p_train.add_argument("--eval", type=Path)
    p_train.add_argument("--run-name", required=True)
    p_train.add_argument("--snapshot", default="")
    p_train.add_argument("--split", choices=("dev", "test"), default=None)
    p_train.add_argument("--target-f1", type=float, default=0.70)
    p_train.add_argument("--by", default="team")
    p_train.add_argument("--notes", default="")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db = QualityTrackerDB(args.database_url)
    db.init_db()

    if args.command == "init":
        print(f"Initialized database schema & default targets on {db.backend_label}.")
        return 0

    if args.command == "scorecard":
        print(f"Backend: {db.backend_label}")
        print(format_scorecard(db.compute_readiness_scorecard()))
        return 0

    if args.command == "iaa":
        payload = json.loads(args.file.read_text(encoding="utf-8"))
        row_id = ingest_iaa_payload(db, payload, batch_name=args.batch, recorded_by=args.by, notes=args.notes)
        print(f"Recorded IAA check #{row_id} on {db.backend_label}.")
        return 0

    if args.command == "qa":
        payload = json.loads(args.file.read_text(encoding="utf-8"))
        row_id = ingest_qa_payload(db, payload, batch_name=args.batch, recorded_by=args.by, notes=args.notes)
        print(f"Recorded Label QA audit #{row_id} on {db.backend_label}.")
        return 0

    if args.command == "split":
        payload = json.loads(args.file.read_text(encoding="utf-8"))
        row_id = ingest_split_payload(
            db,
            payload,
            snapshot_name=args.name,
            sources=args.sources,
            licence=args.licence,
            raw_articles=args.raw_articles,
            cleaned_sentences=args.cleaned_sentences,
            segmented_sentences=args.segmented_sentences,
            recorded_by=args.by,
            notes=args.notes,
        )
        print(f"Recorded dataset snapshot #{row_id} on {db.backend_label}.")
        return 0

    if args.command == "train":
        if not args.summary and not args.eval:
            raise SystemExit("Provide --summary and/or --eval JSON path.")
        summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary else None
        eval_data = json.loads(args.eval.read_text(encoding="utf-8")) if args.eval else None
        row_id = ingest_training_payload(
            db,
            run_name=args.run_name,
            summary_payload=summary,
            eval_payload=eval_data,
            snapshot_name=args.snapshot,
            split_evaluated=args.split,
            target_f1=args.target_f1,
            recorded_by=args.by,
            notes=args.notes,
        )
        print(f"Recorded training run #{row_id} on {db.backend_label}.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
