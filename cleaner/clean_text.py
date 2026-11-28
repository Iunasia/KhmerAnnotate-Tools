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
4. Fix Khmer vowel encoding: a vowel typed as two parts (េ + ី for ើ,
   េ + ា for ោ) becomes the single correct character, and deprecated
   characters are replaced.
5. Fix Khmer typing order: the marks of each syllable are put in the standard
   order (subscripts, register shifter, vowel, final signs), so words that
   look the same are stored the same way.
6. Split paragraphs into sentences at ។ ៕ ៚ ? and !.

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

ZWSP = "\u200b"
JOINERS = "\u200c\u200d"  # zero-width non-joiner, zero-width joiner
KHMER = "\u1780-\u17ff"

# Hidden characters that are always removed (ZWSP is handled separately).
_HIDDEN = (
    "\u00ad"  # soft hyphen
    "\u034f"  # combining grapheme joiner
    "\u061c\u200e\u200f"  # direction marks
    "\u17b4\u17b5"  # Khmer invisible inherent vowels
    "\u202a-\u202e"  # direction embeddings/overrides
    "\u2060-\u2064"  # word joiner, invisible operators
    "\u2066-\u2069"  # direction isolates
    "\ufeff"  # byte-order mark / zero-width no-break space
    "\ufff9-\ufffb"  # interlinear annotation marks
)
_HIDDEN_RE = re.compile(f"[{_HIDDEN}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_STRAY_JOINER_RE = re.compile(f"(?<![{KHMER}])[{JOINERS}]|[{JOINERS}](?![{KHMER}])")
_SPACE_RE = re.compile("[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_NEWLINE_RE = re.compile("\r\n?|[\u2028\u2029\u0085]")
_KHMER_RE = re.compile(f"[{KHMER}]")

# Deprecated Khmer characters and what Unicode says to use instead.
_DEPRECATED = {
    "\u17a3": "\u17a2",  # ឣ -> អ
    "\u17a4": "\u17a2\u17b6",  # ឤ -> អា
    "\u17a8": "\u17a7\u1780",  # ឨ -> ឧក
    "\u17d8": "\u17d4\u179b\u17d4",  # ៘ -> ។ល។
}
_DEPRECATED_RE = re.compile("[" + "".join(_DEPRECATED) + "]")

_BASE = "\u1780-\u17b3"  # consonants and independent vowels
COENG = "\u17d2"  # subscript sign
_MARK = "\u17b6-\u17d1\u17dd"  # dependent vowels and signs
# One syllable: a base character followed by subscripts, marks and joiners.
_CLUSTER_RE = re.compile(f"[{_BASE}](?:{COENG}[{_BASE}]|[{_MARK}{JOINERS}])+")
_UNIT_RE = re.compile(f"{COENG}.|.", re.S)
# Two-part vowels (after ordering, េ comes first; a below-vowel may sit between).
_SPLIT_VOWELS = (
    (re.compile("\u17c1([\u17bb-\u17bd]?)\u17b8"), "\u17be\\1"),  # េ + ី -> ើ
    (re.compile("\u17c1([\u17bb-\u17bd]?)\u17b6"), "\u17c4\\1"),  # េ + ា -> ោ
)


def _unit_order(unit: str) -> float:
    """Position of one part of a syllable in the standard typing order."""
    if unit[0] == COENG:
        return 2.5 if unit[1] == "\u179a" else 2  # subscript រ goes last
    code = ord(unit)
    if code == 0x17CC:  # robat
        return 1
    if code in (0x17C9, 0x17CA):  # register shifters
        return 3
    if unit in JOINERS:
        return 4
    if 0x17C1 <= code <= 0x17C3:  # vowels written before the consonant
        return 5
    if 0x17BB <= code <= 0x17BD:  # vowels below
        return 5.1
    if 0x17B7 <= code <= 0x17BA:  # vowels above
        return 5.2
    if code <= 0x17C5:  # remaining vowels
        return 5.3
    if code in (0x17C7, 0x17C8):  # final signs ះ ៈ
        return 7
    return 6  # other signs (ំ ់ ៍ ...)


def normalise_khmer(text: str, counts: Counter | None = None) -> str:
    """Fix Khmer vowel encoding and typing order. Visible text is unchanged."""

    def count(key: str, number: int = 1) -> None:
        if counts is not None and number:
            counts[key] += number

    def deprecated(match: re.Match) -> str:
        count("deprecated-character")
        return _DEPRECATED[match.group()]

    text = _DEPRECATED_RE.sub(deprecated, text)
    text, number = re.subn(f"{COENG}{{2,}}", COENG, text)
    count("repeated-mark", number)
    # Subscript ដ and subscript ត look identical; store ត, except after ណ.
    text, number = re.subn(f"(?<!\u178e){COENG}\u178a", f"{COENG}\u178f", text)
    text, number2 = re.subn(f"(?<=\u178e){COENG}\u178f", f"{COENG}\u178a", text)
    count("subscript-da-ta", number + number2)

    def cluster(match: re.Match) -> str:
        original = match.group()
        units = _UNIT_RE.findall(original[1:])
        ordered = sorted(units, key=_unit_order)  # stable: ties keep their order
        if ordered != units:
            count("typing-order")
        kept = []
        for unit in ordered:
            # The same mark twice in a row is invisible; េេ is left alone
            # because it is visible.
            if kept and kept[-1] == unit and unit != "\u17c1":
                count("repeated-mark")
                continue
            kept.append(unit)
        marks = "".join(kept)
        for pattern, replacement in _SPLIT_VOWELS:
            marks, number = pattern.subn(replacement, marks)
            count("vowel-encoding", number)
        return original[0] + marks

    return _CLUSTER_RE.sub(cluster, text)


_END = "។៕៚?!"
_CLOSERS = "”\"’'»)\\]"
# A sentence is text up to and including its end marks and any closing quote
# or bracket. "។ល។" (and spaced variants like "។ ល។") is not treated as a sentence end.
_SENTENCE_RE = re.compile(
    f"(?:។\\s*ល\\s*។|[^{_END}])+(?:[{_END}]+[{_CLOSERS}]*)?|[{_END}]+[{_CLOSERS}]*"
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
    text = normalise_khmer(text, counts)
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
    fixes = [
        f"{key} x{counts[key]}"
        for key in ("typing-order", "vowel-encoding", "subscript-da-ta",
                    "repeated-mark", "deprecated-character")
        if counts[key]
    ]
    if fixes:
        parts.append("Khmer fixes: " + ", ".join(fixes))
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
