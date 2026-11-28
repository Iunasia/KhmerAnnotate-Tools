"""Stage 4b: Automated Khmer NER Label Quality Gate.

Audits CoNLL-2002 / JSONL annotated files before curation sign-off or training:
    Errors (must be 0 to pass):
        - Unknown labels outside O, B/I-PER, B/I-LOC, B/I-ORG
        - Invalid BIO transitions (orphan I-X after O or wrong entity type)
        - Broken Khmer character clusters (token starting with coeng or combining mark)
        - Punctuation-only tokens included inside an entity span
        - Empty tokens or empty sentences
    Warnings (review during curation; optional --fail-on-warnings):
        - Hidden ZWSP/BOM/soft-hyphen characters inside tokens
        - Honorific/title (លោក, លោកស្រី, ...) included at the start of a PER span (Rulebook Case 1)
        - Inconsistent entity types for the same surface text across the corpus

Examples (run from the repository root):

    python checker/label_checker.py --input data/annotated/curated.conll --output-dir outputs/qa/labels
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import LABELS, Sentence, load_dataset_file, validate_sentences
from baseline.metrics import extract_spans
from cleaner.segment_words import KHMER_COENG, KHMER_COMBINING_MARKS

# Rulebook Case 1: honorifics/titles that should normally stay outside PER spans.
KHMER_HONORIFICS = frozenset(
    {
        "លោក",
        "លោកស្រី",
        "អ្នកស្រី",
        "អ្នកនាង",
        "កញ្ញា",
        "សម្តេច",
        "សម្ដេច",
        "ឯកឧត្តម",
        "បណ្ឌិត",
        "នាយករដ្ឋមន្ត្រី",
        "ប្រធានាធិបតី",
        "ព្រះមហាក្សត្រ",
        "ឧត្តមនាវី",
    }
)

# Punctuation/quote characters that should never be labeled as part of an entity span.
_PUNCT_ONLY_RE = re.compile(r"^[\u17d4-\u17da«»\"'()\[\]{},.:;?!…\-–—]+$")


@dataclass
class CheckIssue:
    sentence_index: int
    token_index: int | None
    kind: str
    severity: str  # "error" or "warning"
    message: str
    context: str = ""


def audit_sentences(
    sentences: Sequence[Sentence],
    *,
    allowed_labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Run all BIO, Unicode-cluster, rulebook, and consistency checks."""
    issues: list[CheckIssue] = []

    # 1. Core BIO, schema, and hidden-character checks from baseline.data
    for base_issue in validate_sentences(sentences, allowed_labels=allowed_labels):
        sent = sentences[base_issue.sentence_index]
        issues.append(
            CheckIssue(
                sentence_index=base_issue.sentence_index,
                token_index=base_issue.token_index,
                kind=base_issue.kind,
                severity=base_issue.severity,
                message=base_issue.message,
                context=" ".join(sent.tokens[:20]),
            )
        )

    # 2. Khmer-specific cluster, punctuation, honorific, and surface-consistency checks
    surface_types: dict[str, Counter[str]] = defaultdict(Counter)
    surface_locations: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for sent_index, sentence in enumerate(sentences):
        context = " ".join(sentence.tokens[:20])
        for tok_index, (token, tag) in enumerate(zip(sentence.tokens, sentence.tags)):
            if token and (token[0] == KHMER_COENG or token[0] in KHMER_COMBINING_MARKS):
                issues.append(
                    CheckIssue(
                        sentence_index=sent_index,
                        token_index=tok_index,
                        kind="broken-khmer-cluster",
                        severity="error",
                        message=f"token starts with combining mark U+{ord(token[0]):04X}",
                        context=context,
                    )
                )
            if tag != "O" and _PUNCT_ONLY_RE.match(token):
                issues.append(
                    CheckIssue(
                        sentence_index=sent_index,
                        token_index=tok_index,
                        kind="punctuation-in-entity",
                        severity="error",
                        message=f"punctuation token {token!r} labeled as {tag}",
                        context=context,
                    )
                )

        for span in extract_spans(sentence.tags, strict_iob2=True):
            span_tokens = sentence.tokens[span.start : span.end]
            phrase = " ".join(span_tokens)
            surface_types[phrase][span.entity_type] += 1
            surface_locations[phrase].append((sent_index, span.entity_type))

            if span.entity_type == "PER" and span_tokens and span_tokens[0] in KHMER_HONORIFICS:
                issues.append(
                    CheckIssue(
                        sentence_index=sent_index,
                        token_index=span.start,
                        kind="honorific-in-per",
                        severity="warning",
                        message=f"PER span starts with title/honorific ({span_tokens[0]}) — check Rulebook Case 1",
                        context=phrase,
                    )
                )

    inconsistent_entities: list[dict[str, Any]] = []
    for phrase, type_counts in sorted(surface_types.items()):
        if len(type_counts) > 1:
            locs = surface_locations[phrase]
            inconsistent_entities.append(
                {
                    "text": phrase,
                    "type_counts": dict(type_counts),
                    "sentence_indices": sorted({idx for idx, _ in locs}),
                }
            )
            first_sent = locs[0][0]
            summary = ", ".join(f"{etype}={cnt}" for etype, cnt in type_counts.most_common())
            issues.append(
                CheckIssue(
                    sentence_index=first_sent,
                    token_index=None,
                    kind="inconsistent-entity",
                    severity="warning",
                    message=f"entity labeled with multiple types ({summary})",
                    context=phrase,
                )
            )

    errors = [item for item in issues if item.severity == "error"]
    warnings = [item for item in issues if item.severity == "warning"]
    kind_counts = Counter(item.kind for item in issues)

    return {
        "sentences": len(sentences),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "passed": len(errors) == 0,
        "counts_by_kind": dict(kind_counts),
        "inconsistent_entities": inconsistent_entities,
        "issues": [asdict(item) for item in issues],
    }


def write_qa_report_html(report: dict[str, Any], output_path: Path, *, max_rows: int = 200) -> Path:
    """Write a readable UTF-8 HTML quality audit report for Khmer text."""
    rows = []
    for item in report["issues"][:max_rows]:
        sev = item["severity"]
        rows.append(
            f"<tr class='sev-{html.escape(sev)}'>"
            f"<td>{item['sentence_index'] + 1}</td>"
            f"<td>{'' if item['token_index'] is None else item['token_index']}</td>"
            f"<td><strong>{html.escape(sev.upper())}</strong></td>"
            f"<td><code>{html.escape(item['kind'])}</code></td>"
            f"<td>{html.escape(item['message'])}</td>"
            f"<td>{html.escape(item['context'])}</td></tr>"
        )

    table_body = (
        "\n".join(rows)
        if rows
        else "<tr><td colspan='6'>No errors or warnings found. Quality gate passed.</td></tr>"
    )
    status = "PASSED (0 errors)" if report["passed"] else f"FAILED ({report['error_count']} errors)"

    document = f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="utf-8">
  <title>KhmerAnnotate — Label Quality Audit Report</title>
  <style>
    body {{
      font-family: "Noto Sans Khmer", "Khmer OS Battambang", "Segoe UI", sans-serif;
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px;
      line-height: 1.6;
      color: #1f2937;
      background: #f9fafb;
    }}
    h1, h2 {{ font-family: "Segoe UI", sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; font-family: "Segoe UI", sans-serif; }}
    tr.sev-error {{ background: #fef2f2; }}
    tr.sev-warning {{ background: #fffbeb; }}
  </style>
</head>
<body>
  <h1>KhmerAnnotate — Label Quality Audit Report</h1>
  <p>Status: <strong>{html.escape(status)}</strong> |
     Sentences checked: <strong>{report['sentences']}</strong> |
     Errors: <strong>{report['error_count']}</strong> |
     Warnings: <strong>{report['warning_count']}</strong></p>

  <h2>Detected Issues (showing up to {max_rows})</h2>
  <table>
    <thead>
      <tr><th>Sentence #</th><th>Token</th><th>Severity</th><th>Kind</th><th>Message</th><th>Context</th></tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
</body>
</html>
"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit CoNLL-2002 / JSONL Khmer NER annotations for BIO, cluster, and rulebook errors.",
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="one or more CoNLL/JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qa/labels"))
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_sentences: list[Sentence] = []
    for path in args.input:
        if not path.is_file():
            print(f"{path}: not found", file=sys.stderr)
            return 1
        all_sentences.extend(load_dataset_file(path, require_tags=True))

    report = audit_sentences(all_sentences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "qa_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_qa_report_html(report, args.output_dir / "qa_report.html")

    print(
        f"Checked {report['sentences']} sentences: "
        f"{report['error_count']} error(s), {report['warning_count']} warning(s)."
    )
    if report["counts_by_kind"]:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(report["counts_by_kind"].items()))
        print(f"Breakdown: {summary}")
    print(f"Saved {json_path} and {html_path}")

    if report["error_count"] > 0:
        return 1
    if args.fail_on_warnings and report["warning_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
