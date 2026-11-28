"""Exact span-level NER evaluation, error breakdown, and HTML error reporting.

Computes standard CoNLL-2002 / seqeval-compatible entity-level Precision,
Recall, and F1 per entity type (PER, LOC, ORG) and overall micro-F1, plus
an entity confusion matrix, error categorization (exact, type error, boundary
error, missed, spurious), and a UTF-8 HTML error viewer for Khmer text.
"""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from baseline.data import ENTITY_TYPES, Sentence


@dataclass(frozen=True)
class EntitySpan:
    """Half-open token span ``[start, end)`` for one named entity."""

    entity_type: str
    start: int
    end: int

    def overlaps(self, other: EntitySpan) -> bool:
        return self.start < other.end and other.start < self.end


def extract_spans(tags: Sequence[str], *, strict_iob2: bool = False) -> list[EntitySpan]:
    """Extract ``EntitySpan`` objects from a BIO tag sequence.

    With ``strict_iob2=False`` (standard CoNLL-2002 / seqeval default), an
    orphan ``I-X`` starts a new span of type ``X``. With ``strict_iob2=True``,
    orphan ``I-X`` tags are ignored instead of starting a span.
    """
    spans: list[EntitySpan] = []
    current_type: str | None = None
    current_start: int = 0

    def close(end_index: int) -> None:
        nonlocal current_type
        if current_type is not None:
            spans.append(EntitySpan(current_type, current_start, end_index))
            current_type = None

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close(index)
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix == "B":
            close(index)
            current_type = entity_type
            current_start = index
        elif prefix == "I":
            if current_type == entity_type:
                continue
            close(index)
            if not strict_iob2:
                current_type = entity_type
                current_start = index
        else:
            close(index)

    close(len(tags))
    return spans


def _prf1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def compute_ner_metrics(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    *,
    entity_types: Sequence[str] = ENTITY_TYPES,
    strict_iob2: bool = False,
) -> dict[str, Any]:
    """Compute exact span-level Precision, Recall, and F1 per entity and overall."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")

    tp_by_type: Counter[str] = Counter()
    fp_by_type: Counter[str] = Counter()
    fn_by_type: Counter[str] = Counter()

    all_types = list(entity_types)
    seen_extra: set[str] = set()

    for sent_index, (true_tags, pred_tags) in enumerate(zip(y_true, y_pred)):
        if len(true_tags) != len(pred_tags):
            raise ValueError(
                f"Sentence {sent_index}: true length ({len(true_tags)}) != pred length ({len(pred_tags)})"
            )
        gold_spans = set(extract_spans(true_tags, strict_iob2=strict_iob2))
        pred_spans = set(extract_spans(pred_tags, strict_iob2=strict_iob2))

        for span in gold_spans & pred_spans:
            tp_by_type[span.entity_type] += 1
            seen_extra.add(span.entity_type)
        for span in pred_spans - gold_spans:
            fp_by_type[span.entity_type] += 1
            seen_extra.add(span.entity_type)
        for span in gold_spans - pred_spans:
            fn_by_type[span.entity_type] += 1
            seen_extra.add(span.entity_type)

    for extra in sorted(seen_extra):
        if extra not in all_types:
            all_types.append(extra)

    per_entity = {
        etype: _prf1(tp_by_type[etype], fp_by_type[etype], fn_by_type[etype])
        for etype in all_types
    }

    total_tp = sum(tp_by_type.values())
    total_fp = sum(fp_by_type.values())
    total_fn = sum(fn_by_type.values())
    overall = _prf1(total_tp, total_fp, total_fn)

    macro_f1 = (
        sum(per_entity[etype]["f1"] for etype in all_types) / len(all_types)
        if all_types
        else 0.0
    )
    overall["macro_f1"] = round(macro_f1, 6)

    return {
        "overall": overall,
        "per_entity": per_entity,
    }


def analyze_errors(
    sentences: Sequence[Sentence],
    y_pred: Sequence[Sequence[str]],
    *,
    entity_types: Sequence[str] = ENTITY_TYPES,
    strict_iob2: bool = False,
) -> dict[str, Any]:
    """Categorize span-level errors and build an entity confusion matrix.

    Error categories:
        - ``correct``: exact boundary and type match
        - ``type_error``: exact boundary match, wrong entity type
        - ``boundary_error``: overlapping span, same entity type, wrong boundary
        - ``boundary_and_type_error``: overlapping span, wrong boundary and type
        - ``missed``: gold entity with no overlapping prediction (FN)
        - ``spurious``: predicted entity with no overlapping gold entity (FP)
    """
    if len(sentences) != len(y_pred):
        raise ValueError(f"sentences ({len(sentences)}) and y_pred ({len(y_pred)}) length mismatch")

    labels = ["O", *entity_types]
    confusion: dict[str, dict[str, int]] = {
        gold_label: {pred_label: 0 for pred_label in labels}
        for gold_label in labels
    }

    def inc_confusion(gold_label: str, pred_label: str) -> None:
        if gold_label not in confusion:
            confusion[gold_label] = {label: 0 for label in labels}
        if pred_label not in confusion[gold_label]:
            confusion[gold_label][pred_label] = 0
        confusion[gold_label][pred_label] += 1

    counts: Counter[str] = Counter()
    error_examples: list[dict[str, Any]] = []

    for sent_index, (sentence, pred_tags) in enumerate(zip(sentences, y_pred)):
        gold_spans = extract_spans(sentence.tags, strict_iob2=strict_iob2)
        pred_spans = extract_spans(pred_tags, strict_iob2=strict_iob2)

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        sent_errors: list[dict[str, Any]] = []

        # 1. Exact matches (same boundary + same type)
        for g_idx, g_span in enumerate(gold_spans):
            for p_idx, p_span in enumerate(pred_spans):
                if p_idx in matched_pred:
                    continue
                if g_span == p_span:
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    counts["correct"] += 1
                    inc_confusion(g_span.entity_type, p_span.entity_type)
                    break

        # 2. Exact boundary, wrong type
        for g_idx, g_span in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            for p_idx, p_span in enumerate(pred_spans):
                if p_idx in matched_pred:
                    continue
                if g_span.start == p_span.start and g_span.end == p_span.end:
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    counts["type_error"] += 1
                    inc_confusion(g_span.entity_type, p_span.entity_type)
                    sent_errors.append(
                        {
                            "category": "type_error",
                            "gold_text": " ".join(sentence.tokens[g_span.start : g_span.end]),
                            "gold_type": g_span.entity_type,
                            "pred_text": " ".join(sentence.tokens[p_span.start : p_span.end]),
                            "pred_type": p_span.entity_type,
                        }
                    )
                    break

        # 3. Overlapping spans (boundary error or boundary + type error)
        for g_idx, g_span in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            for p_idx, p_span in enumerate(pred_spans):
                if p_idx in matched_pred:
                    continue
                if g_span.overlaps(p_span):
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    category = (
                        "boundary_error"
                        if g_span.entity_type == p_span.entity_type
                        else "boundary_and_type_error"
                    )
                    counts[category] += 1
                    inc_confusion(g_span.entity_type, p_span.entity_type)
                    sent_errors.append(
                        {
                            "category": category,
                            "gold_text": " ".join(sentence.tokens[g_span.start : g_span.end]),
                            "gold_type": g_span.entity_type,
                            "pred_text": " ".join(sentence.tokens[p_span.start : p_span.end]),
                            "pred_type": p_span.entity_type,
                        }
                    )
                    break

        # 4. Missed gold entities (FN)
        for g_idx, g_span in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            counts["missed"] += 1
            inc_confusion(g_span.entity_type, "O")
            sent_errors.append(
                {
                    "category": "missed",
                    "gold_text": " ".join(sentence.tokens[g_span.start : g_span.end]),
                    "gold_type": g_span.entity_type,
                    "pred_text": "",
                    "pred_type": "O",
                }
            )

        # 5. Spurious predicted entities (FP)
        for p_idx, p_span in enumerate(pred_spans):
            if p_idx in matched_pred:
                continue
            counts["spurious"] += 1
            inc_confusion("O", p_span.entity_type)
            sent_errors.append(
                {
                    "category": "spurious",
                    "gold_text": "",
                    "gold_type": "O",
                    "pred_text": " ".join(sentence.tokens[p_span.start : p_span.end]),
                    "pred_type": p_span.entity_type,
                }
            )

        if sent_errors:
            error_examples.append(
                {
                    "sentence_index": sent_index,
                    "doc_id": sentence.doc_id,
                    "sent_id": sentence.sent_id,
                    "tokens": sentence.tokens,
                    "gold_tags": sentence.tags,
                    "pred_tags": list(pred_tags),
                    "errors": sent_errors,
                }
            )

    summary_counts = {
        key: counts.get(key, 0)
        for key in (
            "correct",
            "type_error",
            "boundary_error",
            "boundary_and_type_error",
            "missed",
            "spurious",
        )
    }

    return {
        "counts": summary_counts,
        "confusion_labels": labels,
        "confusion_matrix": confusion,
        "error_sentence_count": len(error_examples),
        "error_examples": error_examples,
    }


def _render_tagged_html(tokens: Sequence[str], tags: Sequence[str]) -> str:
    """Render one sentence with color-coded entity pills for HTML viewing."""
    spans = extract_spans(tags)
    span_by_start = {span.start: span for span in spans}
    parts: list[str] = []
    index = 0
    while index < len(tokens):
        span = span_by_start.get(index)
        if span is not None:
            phrase = html.escape(" ".join(tokens[span.start : span.end]))
            etype = html.escape(span.entity_type)
            parts.append(
                f'<span class="ent ent-{etype}">{phrase}'
                f'<span class="tag">{etype}</span></span>'
            )
            index = span.end
        else:
            parts.append(html.escape(tokens[index]))
            index += 1
    return " ".join(parts)


def write_error_report_html(
    metrics: dict[str, Any],
    error_analysis: dict[str, Any],
    output_path: Path,
    *,
    title: str = "KhmerAnnotate — XLM-R NER Error Report",
    max_examples: int = 100,
) -> Path:
    """Write a readable UTF-8 HTML report for Khmer NER inspection.

    Terminals do not shape Khmer script reliably; viewing predictions in an HTML
    page makes boundary and type errors easy to inspect.
    """
    overall = metrics["overall"]
    per_entity = metrics["per_entity"]
    counts = error_analysis["counts"]

    metric_rows = []
    for etype, stats in per_entity.items():
        metric_rows.append(
            f"<tr><td><strong>{html.escape(etype)}</strong></td>"
            f"<td>{stats['precision']:.4f}</td>"
            f"<td>{stats['recall']:.4f}</td>"
            f"<td><strong>{stats['f1']:.4f}</strong></td>"
            f"<td>{stats['support']}</td></tr>"
        )
    metric_rows.append(
        f"<tr class='overall'><td><strong>Overall (Micro)</strong></td>"
        f"<td>{overall['precision']:.4f}</td>"
        f"<td>{overall['recall']:.4f}</td>"
        f"<td><strong>{overall['f1']:.4f}</strong></td>"
        f"<td>{overall['support']}</td></tr>"
    )

    example_cards = []
    for item in error_analysis["error_examples"][:max_examples]:
        gold_html = _render_tagged_html(item["tokens"], item["gold_tags"])
        pred_html = _render_tagged_html(item["tokens"], item["pred_tags"])
        badges = ", ".join(
            f"{err['category']} ({err['gold_type']} -> {err['pred_type']})"
            for err in item["errors"]
        )
        header = html.escape(
            f"Sentence #{item['sentence_index'] + 1}"
            + (f" | doc: {item['doc_id']}" if item["doc_id"] else "")
            + (f" | id: {item['sent_id']}" if item["sent_id"] else "")
        )
        example_cards.append(
            f"""
            <div class="card">
              <div class="meta">{header} &mdash; <code>{html.escape(badges)}</code></div>
              <div class="line"><span class="label">Gold:</span> {gold_html}</div>
              <div class="line"><span class="label">Pred:</span> {pred_html}</div>
            </div>
            """
        )

    examples_block = (
        "\n".join(example_cards)
        if example_cards
        else "<p>No span-level errors found on this evaluation split.</p>"
    )

    document = f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
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
    tr.overall {{ background: #eff6ff; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
    .pill {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 14px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; }}
    .meta {{ font-size: 0.85em; color: #4b5563; margin-bottom: 6px; font-family: "Segoe UI", sans-serif; }}
    .line {{ margin: 4px 0; font-size: 1.05em; }}
    .label {{ display: inline-block; width: 48px; font-weight: 600; font-family: "Segoe UI", sans-serif; color: #374151; }}
    .ent {{ padding: 2px 6px; border-radius: 4px; font-weight: 500; }}
    .ent-PER {{ background: #dbeafe; border: 1px solid #93c5fd; }}
    .ent-LOC {{ background: #dcfce7; border: 1px solid #86efac; }}
    .ent-ORG {{ background: #fef3c7; border: 1px solid #fcd34d; }}
    .tag {{ font-size: 0.7em; font-weight: 700; margin-left: 4px; opacity: 0.8; font-family: "Segoe UI", sans-serif; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <h2>Per-Entity Performance</h2>
  <table>
    <thead><tr><th>Entity</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>
      {"".join(metric_rows)}
    </tbody>
  </table>

  <h2>Span Error Breakdown</h2>
  <div class="summary">
    <div class="pill">Correct: <strong>{counts['correct']}</strong></div>
    <div class="pill">Type Error: <strong>{counts['type_error']}</strong></div>
    <div class="pill">Boundary Error: <strong>{counts['boundary_error']}</strong></div>
    <div class="pill">Boundary + Type: <strong>{counts['boundary_and_type_error']}</strong></div>
    <div class="pill">Missed (FN): <strong>{counts['missed']}</strong></div>
    <div class="pill">Spurious (FP): <strong>{counts['spurious']}</strong></div>
  </div>

  <h2>Error Examples (showing up to {max_examples} of {error_analysis['error_sentence_count']} sentences)</h2>
  {examples_block}
</body>
</html>
"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Format a compact ASCII summary table suitable for terminal logs."""
    lines = [
        f"{'Entity':<10} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}",
        "-" * 46,
    ]
    for etype, stats in metrics["per_entity"].items():
        lines.append(
            f"{etype:<10} {stats['precision']:>8.4f} {stats['recall']:>8.4f} "
            f"{stats['f1']:>8.4f} {stats['support']:>8d}"
        )
    overall = metrics["overall"]
    lines.append("-" * 46)
    lines.append(
        f"{'OVERALL':<10} {overall['precision']:>8.4f} {overall['recall']:>8.4f} "
        f"{overall['f1']:>8.4f} {overall['support']:>8d}"
    )
    return "\n".join(lines)
