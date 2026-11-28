"""Stage 4a: Inter-Annotator Agreement (IAA) for CoNLL-2002 Khmer NER files.

Computes:
    - Exact span-level pairwise F1 (overall + per-entity PER, LOC, ORG)
    - Token-level Cohen's Kappa across all 7 BIO labels
    - Entity-only Cohen's Kappa (tokens where >= 1 annotator labeled an entity)

Exports ``iaa_metrics.json`` and a UTF-8 HTML side-by-side disagreement viewer
(``iaa_report.html``) so Khmer span disagreements can be read in a browser.

Examples (run from the repository root):

    # Compare two annotators
    python checker/agreement.py \\
        --annotator-a data/annotated/ann1_pilot.conll \\
        --annotator-b data/annotated/ann2_pilot.conll \\
        --output-dir outputs/qa/iaa

    # Compare all pairs across 3+ annotators
    python checker/agreement.py \\
        --files data/annotated/ann1.conll data/annotated/ann2.conll data/annotated/ann3.conll \\
        --output-dir outputs/qa/iaa_all
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import ENTITY_TYPES, Sentence, load_dataset_file
from baseline.metrics import _render_tagged_html, compute_ner_metrics, extract_spans


def cohens_kappa(tags_a: Sequence[str], tags_b: Sequence[str]) -> float:
    """Compute token-level Cohen's Kappa between two aligned tag sequences."""
    if len(tags_a) != len(tags_b):
        raise ValueError(f"Tag sequences differ in length ({len(tags_a)} vs {len(tags_b)})")
    total = len(tags_a)
    if total == 0:
        return 1.0

    agree = sum(1 for a, b in zip(tags_a, tags_b) if a == b)
    p_o = agree / total

    counts_a = Counter(tags_a)
    counts_b = Counter(tags_b)
    all_labels = set(counts_a) | set(counts_b)
    p_e = sum((counts_a[label] / total) * (counts_b[label] / total) for label in all_labels)

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def compare_annotators(
    sents_a: Sequence[Sentence],
    sents_b: Sequence[Sentence],
    *,
    name_a: str = "Annotator A",
    name_b: str = "Annotator B",
) -> dict[str, Any]:
    """Compute span-level F1, Cohen's Kappa, and sentence-level disagreements."""
    if len(sents_a) != len(sents_b):
        raise ValueError(
            f"Sentence count mismatch: {name_a} has {len(sents_a)}, {name_b} has {len(sents_b)}"
        )

    flat_a: list[str] = []
    flat_b: list[str] = []
    entity_a: list[str] = []
    entity_b: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for index, (sa, sb) in enumerate(zip(sents_a, sents_b)):
        if sa.tokens != sb.tokens:
            raise ValueError(
                f"Token mismatch on sentence #{index + 1} between {name_a} and {name_b}: "
                f"{len(sa.tokens)} vs {len(sb.tokens)} tokens"
            )
        flat_a.extend(sa.tags)
        flat_b.extend(sb.tags)
        for ta, tb in zip(sa.tags, sb.tags):
            if ta != "O" or tb != "O":
                entity_a.append(ta)
                entity_b.append(tb)

        spans_a = extract_spans(sa.tags)
        spans_b = extract_spans(sb.tags)
        if spans_a != spans_b:
            disagreements.append(
                {
                    "sentence_index": index,
                    "doc_id": sa.doc_id or sb.doc_id,
                    "sent_id": sa.sent_id or sb.sent_id,
                    "tokens": sa.tokens,
                    "tags_a": sa.tags,
                    "tags_b": sb.tags,
                    "spans_a": [
                        {"type": sp.entity_type, "text": " ".join(sa.tokens[sp.start : sp.end])}
                        for sp in spans_a
                    ],
                    "spans_b": [
                        {"type": sp.entity_type, "text": " ".join(sb.tokens[sp.start : sp.end])}
                        for sp in spans_b
                    ],
                }
            )

    # Span F1 is symmetric: swapping gold/pred swaps precision and recall, F1 is identical.
    span_metrics = compute_ner_metrics([s.tags for s in sents_a], [s.tags for s in sents_b])
    token_kappa = round(cohens_kappa(flat_a, flat_b), 6)
    entity_only_kappa = round(cohens_kappa(entity_a, entity_b), 6) if entity_a else 1.0

    return {
        "annotator_a": name_a,
        "annotator_b": name_b,
        "sentences": len(sents_a),
        "tokens": len(flat_a),
        "token_cohens_kappa": token_kappa,
        "entity_only_kappa": entity_only_kappa,
        "span_f1": span_metrics["overall"]["f1"],
        "per_entity_span_f1": {
            etype: span_metrics["per_entity"][etype]["f1"]
            for etype in ENTITY_TYPES
            if etype in span_metrics["per_entity"]
        },
        "disagreement_sentences": len(disagreements),
        "disagreements": disagreements,
    }


def write_iaa_report_html(
    report: dict[str, Any],
    output_path: Path,
    *,
    target_iaa: float = 0.80,
    max_examples: int = 100,
) -> Path:
    """Write a readable UTF-8 HTML disagreement viewer for Khmer IAA review."""
    pair_rows = []
    for pair in report["pairs"]:
        status = "PASS" if pair["span_f1"] >= target_iaa else "BELOW TARGET"
        pair_rows.append(
            f"<tr><td>{html.escape(pair['annotator_a'])} vs {html.escape(pair['annotator_b'])}</td>"
            f"<td><strong>{pair['span_f1']:.4f}</strong></td>"
            f"<td>{pair['token_cohens_kappa']:.4f}</td>"
            f"<td>{pair['entity_only_kappa']:.4f}</td>"
            f"<td>{pair['disagreement_sentences']} / {pair['sentences']}</td>"
            f"<td><strong>{status}</strong></td></tr>"
        )

    cards = []
    shown = 0
    for pair in report["pairs"]:
        for item in pair["disagreements"]:
            if shown >= max_examples:
                break
            shown += 1
            html_a = _render_tagged_html(item["tokens"], item["tags_a"])
            html_b = _render_tagged_html(item["tokens"], item["tags_b"])
            header = html.escape(
                f"{pair['annotator_a']} vs {pair['annotator_b']} | "
                f"Sentence #{item['sentence_index'] + 1}"
                + (f" | doc: {item['doc_id']}" if item["doc_id"] else "")
            )
            cards.append(
                f"""
                <div class="card">
                  <div class="meta">{header}</div>
                  <div class="line"><span class="label">{html.escape(pair['annotator_a'])}:</span> {html_a}</div>
                  <div class="line"><span class="label">{html.escape(pair['annotator_b'])}:</span> {html_b}</div>
                </div>
                """
            )

    examples_html = "\n".join(cards) if cards else "<p>Zero span disagreements found.</p>"
    document = f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="utf-8">
  <title>KhmerAnnotate — Inter-Annotator Agreement (IAA) Report</title>
  <style>
    body {{
      font-family: "Noto Sans Khmer", "Khmer OS Battambang", "Segoe UI", sans-serif;
      max-width: 1000px;
      margin: 24px auto;
      padding: 0 16px;
      line-height: 1.7;
      color: #1f2937;
      background: #f9fafb;
    }}
    h1, h2 {{ font-family: "Segoe UI", sans-serif; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; }}
    .meta {{ font-size: 0.85em; color: #4b5563; margin-bottom: 6px; font-family: "Segoe UI", sans-serif; }}
    .line {{ margin: 4px 0; font-size: 1.05em; }}
    .label {{ display: inline-block; min-width: 110px; font-weight: 600; font-family: "Segoe UI", sans-serif; }}
    .ent {{ padding: 2px 6px; border-radius: 4px; font-weight: 500; }}
    .ent-PER {{ background: #dbeafe; border: 1px solid #93c5fd; }}
    .ent-LOC {{ background: #dcfce7; border: 1px solid #86efac; }}
    .ent-ORG {{ background: #fef3c7; border: 1px solid #fcd34d; }}
    .tag {{ font-size: 0.7em; font-weight: 700; margin-left: 4px; opacity: 0.8; font-family: "Segoe UI", sans-serif; }}
  </style>
</head>
<body>
  <h1>KhmerAnnotate — Inter-Annotator Agreement (IAA) Report</h1>
  <p>Mean Span F1: <strong>{report['mean_span_f1']:.4f}</strong> |
     Mean Token Cohen's Kappa: <strong>{report['mean_token_kappa']:.4f}</strong> |
     Target: <strong>{target_iaa:.2f}</strong></p>

  <h2>Pairwise Agreement</h2>
  <table>
    <thead>
      <tr><th>Pair</th><th>Span F1</th><th>Token Kappa</th><th>Entity-Only Kappa</th><th>Disagreements</th><th>Status</th></tr>
    </thead>
    <tbody>
      {"".join(pair_rows)}
    </tbody>
  </table>

  <h2>Disagreement Examples (showing {shown})</h2>
  {examples_html}
</body>
</html>
"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Inter-Annotator Agreement (Cohen's Kappa & span F1) on CoNLL-2002 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotator-a", type=Path, help="first annotator's CoNLL/JSONL file")
    parser.add_argument("--annotator-b", type=Path, help="second annotator's CoNLL/JSONL file")
    parser.add_argument("--files", type=Path, nargs="+", help="2 or more annotator files for all-pairs IAA")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qa/iaa"))
    parser.add_argument("--target-iaa", type=float, default=0.80)
    parser.add_argument("--strict", action="store_true", help="exit non-zero if mean span F1 < --target-iaa")
    parser.add_argument("--max-html-examples", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.files:
        paths = list(args.files)
    elif args.annotator_a and args.annotator_b:
        paths = [args.annotator_a, args.annotator_b]
    else:
        raise SystemExit("Provide either --annotator-a and --annotator-b, or --files <f1> <f2> ...")

    if len(paths) < 2:
        raise SystemExit("At least 2 annotator files are required.")

    loaded = [(path.stem, load_dataset_file(path, require_tags=True)) for path in paths]
    pairs = [
        compare_annotators(sents_a, sents_b, name_a=name_a, name_b=name_b)
        for (name_a, sents_a), (name_b, sents_b) in itertools.combinations(loaded, 2)
    ]

    mean_span_f1 = round(sum(p["span_f1"] for p in pairs) / len(pairs), 6)
    mean_token_kappa = round(sum(p["token_cohens_kappa"] for p in pairs) / len(pairs), 6)
    mean_entity_kappa = round(sum(p["entity_only_kappa"] for p in pairs) / len(pairs), 6)

    report = {
        "target_iaa": args.target_iaa,
        "passed": mean_span_f1 >= args.target_iaa,
        "mean_span_f1": mean_span_f1,
        "mean_token_kappa": mean_token_kappa,
        "mean_entity_only_kappa": mean_entity_kappa,
        "pairs": pairs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        **report,
        "pairs": [{k: v for k, v in pair.items() if k != "disagreements"} for pair in pairs],
    }
    json_path = args.output_dir / "iaa_metrics.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_iaa_report_html(
        report,
        args.output_dir / "iaa_report.html",
        target_iaa=args.target_iaa,
        max_examples=args.max_html_examples,
    )

    for pair in pairs:
        print(
            f"{pair['annotator_a']} vs {pair['annotator_b']}: "
            f"span_F1={pair['span_f1']:.4f}, token_kappa={pair['token_cohens_kappa']:.4f}, "
            f"entity_kappa={pair['entity_only_kappa']:.4f}, "
            f"disagreements={pair['disagreement_sentences']}/{pair['sentences']}"
        )
    print(f"Mean Span F1: {mean_span_f1:.4f} (target >= {args.target_iaa:.2f})")
    print(f"Saved {json_path} and {html_path}")

    if args.strict and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
