"""Stage 4c: Re-attach provenance metadata and create leak-free 80/10/10 CoNLL-2002 splits.

When INCEpTION exports ``CoNLL-2002`` from an imported ``.txt`` file, the
exported file does not include article ``doc_id`` or ``url`` headers. This
script matches exported sentences back to their Stage 2b ``.jsonl`` metadata
file(s), runs the Stage 4b quality gate, splits by document ID with a fixed
seed so sentences from the same article never appear in both Train and Dev/Test,
verifies zero leakage, and writes ``train.conll``, ``dev.conll``, ``test.conll``
ready for ``baseline/train_xlmr.py``.

Examples (run from the repository root):

    python checker/prepare_splits.py \\
        --conll data/annotated/curated.conll \\
        --metadata data/annotation_batches/annotation_pool_2000.jsonl \\
        --output-dir data/curated
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import (
    Sentence,
    dataset_summary,
    load_dataset_file,
    repair_bio_tags,
    split_by_document,
    verify_no_leakage,
    write_conll,
)
from checker.label_checker import audit_sentences
from cleaner.clean_text import read_jsonl


def _token_key(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def attach_provenance(
    sentences: Sequence[Sentence],
    metadata_records: Sequence[dict[str, Any]],
    *,
    align_by_order: bool = False,
) -> tuple[list[Sentence], dict[str, int]]:
    """Attach ``doc_id``, ``sent_id``, ``url``, ``source``, and ``licence`` to CoNLL sentences."""
    by_tokens: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for rec in metadata_records:
        tokens = rec.get("tokens")
        if not tokens and isinstance(rec.get("segmented_text"), str):
            tokens = rec["segmented_text"].split()
        if tokens:
            by_tokens[_token_key(tokens)].append(rec)

    enriched: list[Sentence] = []
    matched = 0
    already_had_doc = 0
    unmatched = 0

    for index, sentence in enumerate(sentences):
        key = _token_key(sentence.tokens)
        rec = by_tokens[key].popleft() if by_tokens.get(key) else None
        if rec is None and align_by_order and index < len(metadata_records):
            rec = metadata_records[index]

        if rec is not None:
            matched += 1
            doc_id = str(rec.get("doc_id") or rec.get("url") or sentence.doc_id or "")
            sent_id = str(rec.get("id") or rec.get("sentence_id") or sentence.sent_id or "")
            meta = dict(sentence.meta)
            for field in ("url", "source", "title", "licence", "license", "published_date", "attribution"):
                if rec.get(field):
                    meta[field] = str(rec[field])
            enriched.append(
                Sentence(
                    tokens=list(sentence.tokens),
                    tags=list(sentence.tags),
                    doc_id=doc_id,
                    sent_id=sent_id,
                    meta=meta,
                )
            )
        else:
            if sentence.group_key:
                already_had_doc += 1
            else:
                unmatched += 1
            enriched.append(sentence)

    stats = {
        "total": len(sentences),
        "matched_metadata": matched,
        "already_had_doc_id": already_had_doc,
        "unmatched": unmatched,
    }
    return enriched, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-attach document metadata, audit labels, and write leak-free train/dev/test CoNLL splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--conll", type=Path, nargs="+", required=True, help="curated CoNLL-2002 / JSONL file(s)")
    parser.add_argument(
        "--metadata",
        type=Path,
        nargs="*",
        default=[],
        help="Stage 2b segmented .jsonl file(s) containing doc_id/url provenance",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--align-by-order", action="store_true", help="fallback to line-order metadata matching")
    parser.add_argument("--require-metadata", action="store_true", help="fail if any sentence lacks doc_id/url")
    parser.add_argument("--repair-bio", action="store_true", help="auto-convert orphan I-X to B-X before audit")
    parser.add_argument("--allow-errors", action="store_true", help="continue even if label_checker reports errors")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sentences: list[Sentence] = []
    for path in args.conll:
        if not path.is_file():
            print(f"{path}: not found", file=sys.stderr)
            return 1
        sentences.extend(load_dataset_file(path, require_tags=True))

    if not sentences:
        print("Input CoNLL dataset is empty.", file=sys.stderr)
        return 1

    if args.repair_bio:
        for sentence in sentences:
            sentence.tags = repair_bio_tags(sentence.tags)

    meta_records: list[dict[str, Any]] = []
    for meta_path in args.metadata:
        if not meta_path.is_file():
            print(f"{meta_path}: not found", file=sys.stderr)
            return 1
        meta_records.extend(read_jsonl(meta_path))

    if meta_records:
        sentences, provenance_stats = attach_provenance(
            sentences, meta_records, align_by_order=args.align_by_order
        )
    else:
        have_doc = sum(1 for s in sentences if s.group_key)
        provenance_stats = {
            "total": len(sentences),
            "matched_metadata": 0,
            "already_had_doc_id": have_doc,
            "unmatched": len(sentences) - have_doc,
        }

    if args.require_metadata and provenance_stats["unmatched"] > 0:
        print(
            f"Metadata required: {provenance_stats['unmatched']} sentence(s) still lack doc_id/url.",
            file=sys.stderr,
        )
        return 1

    qa_report = audit_sentences(sentences)
    if qa_report["error_count"] > 0 and not args.allow_errors:
        print(
            f"Quality gate failed: {qa_report['error_count']} error(s) found "
            f"({qa_report['counts_by_kind']}). Run checker/label_checker.py for full details.",
            file=sys.stderr,
        )
        return 1

    train_sents, dev_sents, test_sents = split_by_document(
        sentences,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    leakage = verify_no_leakage(train_sents, dev_sents, test_sents)
    if leakage["has_doc_leakage"]:
        print(f"Unexpected document leakage detected: {leakage['doc_overlap']}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_conll(train_sents, args.output_dir / "train.conll")
    write_conll(dev_sents, args.output_dir / "dev.conll")
    write_conll(test_sents, args.output_dir / "test.conll")

    report = {
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "dev": args.dev_ratio, "test": args.test_ratio},
        "provenance": provenance_stats,
        "qa": {
            "error_count": qa_report["error_count"],
            "warning_count": qa_report["warning_count"],
            "counts_by_kind": qa_report["counts_by_kind"],
        },
        "leakage": leakage,
        "splits": {
            "train": dataset_summary(train_sents),
            "dev": dataset_summary(dev_sents),
            "test": dataset_summary(test_sents),
        },
    }
    report_path = args.output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Prepared leak-free splits in {args.output_dir}/: "
        f"train={len(train_sents)}, dev={len(dev_sents)}, test={len(test_sents)} "
        f"(provenance matched={provenance_stats['matched_metadata']}, "
        f"qa_errors={qa_report['error_count']}, qa_warnings={qa_report['warning_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
