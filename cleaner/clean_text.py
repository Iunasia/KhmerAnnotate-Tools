"""Clean raw Khmer text and split it into sentences for annotation.

Reads the collectors' JSONL files (one article per line) and writes one
sentence per line, ready to be loaded into an annotation tool.

Cleaning steps:

1. Unicode NFC normalisation (never NFKC, which rewrites characters).
2. Remove hidden characters: zero-width spaces, BOMs, soft hyphens,
   direction marks and control characters. Zero-width (non-)joiners are kept
   only between two Khmer characters, where they change how text is drawn.
3. Turn unusual spaces (no-break, thin, ideographic, tabs, ...) into a
   normal space and collapse repeated spaces.
4. Split paragraphs into sentences at ។ ៕ ៚ ? and !.

Examples (run from the repository root):

    python cleaner/clean_text.py
    python cleaner/clean_text.py --input data/raw/voa_khmer/voa_khmer.jsonl
    python cleaner/clean_text.py --keep-zwsp --min-khmer 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

ZWSP = "​"
JOINERS = "‌‍"  # zero-width non-joiner, zero-width joiner
KHMER = "ក-៿"

# Hidden characters that are always removed (ZWSP is handled separately).
_HIDDEN = (
    "­"  # soft hyphen
    "͏"  # combining grapheme joiner
    "؜‎‏"  # direction marks
    "‪-‮"  # direction embeddings/overrides
    "⁠-⁤"  # word joiner, invisible operators
    "⁦-⁩"  # direction isolates
    "﻿"  # byte-order mark / zero-width no-break space
    "￹-￻"  # interlinear annotation marks
)
_HIDDEN_RE = re.compile(f"[{_HIDDEN}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_STRAY_JOINER_RE = re.compile(f"(?<![{KHMER}])[{JOINERS}]|[{JOINERS}](?![{KHMER}])")
_SPACE_RE = re.compile("[\t   -   　]")
_NEWLINE_RE = re.compile("\r\n?|[  \u0085]")
_KHMER_RE = re.compile(f"[{KHMER}]")

_END = "។៕៚?!"
_CLOSERS = "”\"’'»)\\]"
# A sentence is text up to and including its end marks and any closing quote
# or bracket. "។ល។" (the Khmer "etc.") is not treated as a sentence end.
_SENTENCE_RE = re.compile(
    f"(?:។ល។|[^{_END}])+(?:[{_END}]+[{_CLOSERS}]*)?|[{_END}]+[{_CLOSERS}]*"
)


def clean_text(text: str, keep_zwsp: bool = False, counts: Counter | None = None) -> str:
    """Return ``text`` with hidden characters removed and spacing tidied.

    Newlines are kept (they separate paragraphs). If ``counts`` is given, the
    number of each removed or replaced character is added to it.
    """
    text = unicodedata.normalize("NFC", text)
    text = _NEWLINE_RE.sub("\n", text)

    def drop(match: re.Match) -> str:
        if counts is not None:
            counts[match.group()] += 1
        return ""

    def to_space(match: re.Match) -> str:
        if counts is not None:
            counts[match.group()] += 1
        return " "

    text = _HIDDEN_RE.sub(drop, text)
    text = _CONTROL_RE.sub(drop, text)
    text = _SPACE_RE.sub(to_space, text)
    if keep_zwsp:
        # Keep one ZWSP as a word-boundary hint, but never next to a space.
        text = re.sub(f"{ZWSP}{{2,}}", ZWSP, text)
        text = re.sub(f"{ZWSP}*([ \n]){ZWSP}*", r"\1", text)
    else:
        if counts is not None:
            counts[ZWSP] += text.count(ZWSP)
        text = text.replace(ZWSP, "")
    # Run until stable: removing one stray joiner can expose another.
    while _STRAY_JOINER_RE.search(text):
        text = _STRAY_JOINER_RE.sub(drop, text)
    text = re.sub(" {2,}", " ", text)
    text = re.sub(" *\n *", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    return text.strip(f" \n{ZWSP}")


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences. Each line is split separately."""
    sentences = []
    for line in text.split("\n"):
        for match in _SENTENCE_RE.finditer(line):
            sentence = match.group().strip(f" {ZWSP}")
            if not sentence:
                continue
            # End marks on their own (e.g. after a space) belong to the
            # previous sentence.
            if sentences and not sentence.strip(_END + _CLOSERS + " "):
                sentences[-1] += sentence
            else:
                sentences.append(sentence)
    return sentences


def khmer_chars(text: str) -> int:
    return len(_KHMER_RE.findall(text))


def sentence_records(
    record: dict,
    prefix: str,
    keep_zwsp: bool = False,
    min_khmer: int = 10,
    counts: Counter | None = None,
) -> Iterator[dict]:
    """Yield one output record per kept sentence of a raw article record."""
    doc_id = str(record.get("id") or record.get("url") or "")
    text = clean_text(record.get("text") or "", keep_zwsp, counts)
    index = 0
    for sentence in split_sentences(text):
        if khmer_chars(sentence) < min_khmer:
            if counts is not None:
                counts["short-sentence"] += 1
            continue
        index += 1
        out = {
            "id": f"{prefix}-{doc_id}-{index:04d}",
            "doc_id": doc_id,
            "sentence_index": index,
            "text": sentence,
        }
        for field in ("source", "url", "title", "licence", "attribution"):
            if record.get(field):
                out[field] = record[field]
        # The text now differs from the publisher's (cleaned and split);
        # CC BY-SA sources require this to be stated.
        out["changed"] = True
        yield out


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                print(f"  {path}:{number}: skipped bad JSON ({error})", file=sys.stderr)


def clean_file(source: Path, target: Path, keep_zwsp: bool, min_khmer: int) -> Counter:
    counts: Counter = Counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as out:
        for record in read_jsonl(source):
            counts["articles"] += 1
            for sentence in sentence_records(record, source.stem, keep_zwsp, min_khmer, counts):
                counts["sentences"] += 1
                out.write(json.dumps(sentence, ensure_ascii=False) + "\n")
    return counts


def find_inputs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("*/*.jsonl") if p.name != "skipped.jsonl")


def describe(counts: Counter) -> str:
    parts = [f"{counts['articles']} articles -> {counts['sentences']} sentences"]
    if counts["short-sentence"]:
        parts.append(f"{counts['short-sentence']} short sentences dropped")
    removed = [
        f"U+{ord(key):04X} x{value}"
        for key, value in counts.most_common()
        if len(key) == 1
    ]
    if removed:
        parts.append("characters removed/replaced: " + ", ".join(removed))
    return "; ".join(parts)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input", type=Path, nargs="*",
        help=f"raw JSONL files (default: every {RAW_DIR.as_posix()}/*/*.jsonl)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--keep-zwsp", action="store_true",
        help="keep zero-width spaces (word-boundary hints) inside sentences",
    )
    parser.add_argument(
        "--min-khmer", type=int, default=10,
        help="drop sentences with fewer Khmer characters than this (default: 10)",
    )
    args = parser.parse_args(argv)

    inputs = args.input or find_inputs(RAW_DIR)
    if not inputs:
        print(f"No raw JSONL files found in {RAW_DIR}/. Run a collector first.", file=sys.stderr)
        return 1
    for source in inputs:
        if not source.is_file():
            print(f"{source}: not found", file=sys.stderr)
            return 1
        target = args.output_dir / f"{source.stem}_sentences.jsonl"
        counts = clean_file(source, target, args.keep_zwsp, args.min_khmer)
        print(f"{source} -> {target}\n  {describe(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
