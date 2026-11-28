"""CoNLL-2002 / JSONL dataset I/O, BIO validation, splitting, and subword alignment.

Standard label set (flat whole-word IOB2):
    O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG

Supports both:
- CoNLL-2002 files (``word<TAB>TAG`` or ``word TAG``, blank lines between
  sentences, optional ``# key = value`` metadata comments such as ``# doc_id``
  or ``# url``, and ``-DOCSTART-`` markers).
- Pre-tokenized JSONL files (``{"tokens": [...], "tags": [...], "doc_id": ...}``).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ENTITY_TYPES: tuple[str, ...] = ("PER", "LOC", "ORG")
LABELS: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
)
LABEL2ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {index: label for index, label in enumerate(LABELS)}
IGNORE_INDEX: int = -100

# Hidden characters that should already be stripped before annotation/training.
_INVISIBLE_WARN_RE = re.compile(r"[\u200b\ufeff\u00ad\u17b4\u17b5]")
_COMMENT_KV_RE = re.compile(r"^#\s*([A-Za-z0-9_ .-]+?)\s*[:=]\s*(.+?)\s*$")


@dataclass
class Sentence:
    """One word-segmented sentence with aligned BIO tags and provenance metadata."""

    tokens: list[str]
    tags: list[str]
    doc_id: str = ""
    sent_id: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def group_key(self) -> str:
        """Stable document identifier used to prevent train/dev/test leakage."""
        return (
            self.doc_id
            or self.meta.get("doc_id")
            or self.meta.get("url")
            or self.sent_id
            or ""
        )


@dataclass
class ValidationIssue:
    """One BIO, schema, or encoding problem detected in a sentence."""

    sentence_index: int
    token_index: int | None
    kind: str
    message: str
    severity: str = "error"  # "error" or "warning"


def read_conll(
    path: Path,
    *,
    word_col: int = 0,
    tag_col: int = -1,
    require_tags: bool = True,
) -> list[Sentence]:
    """Read sentences from a CoNLL-2002 / CoNLL-2003 file.

    Lines starting with ``#`` are parsed for ``# key = value`` metadata (such
    as ``doc_id``, ``url``, ``sent_id``). ``-DOCSTART-`` lines mark document
    boundaries without adding tokens.
    """
    sentences: list[Sentence] = []
    tokens: list[str] = []
    tags: list[str] = []
    meta: dict[str, str] = {}
    current_doc_id = ""

    def flush() -> None:
        nonlocal tokens, tags, meta
        if not tokens:
            return
        doc_id = meta.get("doc_id") or meta.get("url") or current_doc_id
        sent_id = meta.get("sent_id") or meta.get("id") or ""
        sentences.append(
            Sentence(
                tokens=tokens,
                tags=tags,
                doc_id=doc_id,
                sent_id=sent_id,
                meta=dict(meta),
            )
        )
        tokens = []
        tags = []
        meta = {}

    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line.startswith("#"):
                match = _COMMENT_KV_RE.match(line)
                if match:
                    key, value = match.group(1).strip().lower(), match.group(2).strip()
                    meta[key] = value
                    if key in ("doc_id", "newdoc id", "document"):
                        current_doc_id = value
                    elif key == "url" and "doc_id" not in meta:
                        current_doc_id = value
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if parts[0] == "-DOCSTART-":
                flush()
                current_doc_id = f"doc-{len(sentences) + 1}"
                continue

            if require_tags and len(parts) < 2:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 2 columns (token and tag), got {line!r}"
                )
            try:
                token = parts[word_col]
            except IndexError as exc:
                raise ValueError(
                    f"{path}:{line_number}: word column {word_col} out of range for {parts!r}"
                ) from exc

            if require_tags or len(parts) >= 2:
                try:
                    tag = parts[tag_col]
                except IndexError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: tag column {tag_col} out of range for {parts!r}"
                    ) from exc
            else:
                tag = "O"

            tokens.append(token)
            tags.append(tag)

    flush()
    return sentences


def write_conll(
    sentences: Sequence[Sentence],
    path: Path,
    *,
    predicted_tags: Sequence[Sequence[str]] | None = None,
    include_meta: bool = True,
) -> None:
    """Write sentences in CoNLL-2002 format (tab-separated).

    If ``predicted_tags`` is provided, writes three columns:
    ``token<TAB>gold_tag<TAB>pred_tag``.
    """
    if predicted_tags is not None and len(predicted_tags) != len(sentences):
        raise ValueError(
            f"predicted_tags length ({len(predicted_tags)}) != sentences length ({len(sentences)})"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sentence in enumerate(sentences):
            if include_meta:
                if sentence.doc_id:
                    handle.write(f"# doc_id = {sentence.doc_id}\n")
                if sentence.sent_id:
                    handle.write(f"# sent_id = {sentence.sent_id}\n")
                for key, value in sentence.meta.items():
                    if key not in ("doc_id", "sent_id"):
                        handle.write(f"# {key} = {value}\n")

            preds = predicted_tags[index] if predicted_tags is not None else None
            if preds is not None and len(preds) != len(sentence.tokens):
                raise ValueError(
                    f"Sentence {index}: predicted length ({len(preds)}) != token length ({len(sentence.tokens)})"
                )
            for token_index, (token, gold) in enumerate(zip(sentence.tokens, sentence.tags)):
                if preds is not None:
                    handle.write(f"{token}\t{gold}\t{preds[token_index]}\n")
                else:
                    handle.write(f"{token}\t{gold}\n")
            handle.write("\n")


def read_jsonl_dataset(path: Path, *, require_tags: bool = True) -> list[Sentence]:
    """Read pre-tokenized NER records from a JSONL file.

    Expected keys per line:
        ``tokens`` (list[str]) or ``text`` (space-segmented str),
        ``tags`` or ``ner_tags`` (list[str], optional when ``require_tags=False``),
        optional ``doc_id`` / ``url`` / ``id``.
    """
    sentences: list[Sentence] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc

            tokens = record.get("tokens")
            if tokens is None and isinstance(record.get("text"), str):
                tokens = record["text"].split()
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"{path}:{line_number}: missing non-empty 'tokens'")

            tags = record.get("tags") if record.get("tags") is not None else record.get("ner_tags")
            if tags is None:
                if require_tags:
                    raise ValueError(f"{path}:{line_number}: missing 'tags' or 'ner_tags'")
                tags = ["O"] * len(tokens)
            elif len(tags) != len(tokens):
                raise ValueError(
                    f"{path}:{line_number}: tokens ({len(tokens)}) and tags ({len(tags)}) length mismatch"
                )

            meta = {
                key: str(record[key])
                for key in ("source", "url", "title", "licence", "license", "published_date")
                if record.get(key)
            }
            doc_id = str(record.get("doc_id") or record.get("url") or "")
            sent_id = str(record.get("sent_id") or record.get("id") or "")
            sentences.append(
                Sentence(
                    tokens=[str(t) for t in tokens],
                    tags=[str(tag) for tag in tags],
                    doc_id=doc_id,
                    sent_id=sent_id,
                    meta=meta,
                )
            )
    return sentences


def load_dataset_file(path: Path, *, require_tags: bool = True) -> list[Sentence]:
    """Load either a ``.jsonl`` or CoNLL-2002 (``.conll``, ``.txt``, ``.bio``) file."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_dataset(path, require_tags=require_tags)
    return read_conll(path, require_tags=require_tags)


def validate_sentences(
    sentences: Sequence[Sentence],
    *,
    allowed_labels: Sequence[str] = LABELS,
) -> list[ValidationIssue]:
    """Check BIO transition validity, unknown labels, and hidden Unicode characters."""
    allowed = set(allowed_labels)
    issues: list[ValidationIssue] = []

    for sent_index, sentence in enumerate(sentences):
        if not sentence.tokens:
            issues.append(ValidationIssue(sent_index, None, "empty-sentence", "sentence has no tokens"))
            continue
        if len(sentence.tokens) != len(sentence.tags):
            issues.append(
                ValidationIssue(
                    sent_index,
                    None,
                    "length-mismatch",
                    f"{len(sentence.tokens)} tokens != {len(sentence.tags)} tags",
                )
            )
            continue

        prev_tag = "O"
        for tok_index, (token, tag) in enumerate(zip(sentence.tokens, sentence.tags)):
            if not token.strip():
                issues.append(ValidationIssue(sent_index, tok_index, "empty-token", "blank token"))
            if _INVISIBLE_WARN_RE.search(token):
                issues.append(
                    ValidationIssue(
                        sent_index,
                        tok_index,
                        "invisible-unicode",
                        "token contains ZWSP/BOM/soft-hyphen; run cleaner first",
                        severity="warning",
                    )
                )
            if tag not in allowed:
                issues.append(
                    ValidationIssue(sent_index, tok_index, "unknown-label", f"unknown label {tag!r}")
                )
                prev_tag = "O"
                continue
            if tag.startswith("I-"):
                entity_type = tag[2:]
                valid_prev = (f"B-{entity_type}", f"I-{entity_type}")
                if prev_tag not in valid_prev:
                    issues.append(
                        ValidationIssue(
                            sent_index,
                            tok_index,
                            "invalid-bio",
                            f"orphan {tag} after {prev_tag}",
                        )
                    )
            prev_tag = tag

    return issues


def repair_bio_tags(tags: Sequence[str]) -> list[str]:
    """Convert any orphan ``I-X`` tag into ``B-X`` while leaving valid tags intact."""
    fixed: list[str] = []
    prev_tag = "O"
    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if prev_tag not in (f"B-{entity_type}", f"I-{entity_type}"):
                tag = f"B-{entity_type}"
        fixed.append(tag)
        prev_tag = tag
    return fixed


def split_by_document(
    sentences: Sequence[Sentence],
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    """Split sentences into train/dev/test grouped by ``doc_id`` (zero document leakage).

    Sentences without a ``doc_id``/``url`` are each placed in their own fallback
    group using their sentence index.
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}")

    groups: dict[str, list[Sentence]] = defaultdict(list)
    for index, sentence in enumerate(sentences):
        key = sentence.group_key or f"__ungrouped_{index}"
        groups[key].append(sentence)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    total_sentences = len(sentences)
    targets = {
        "train": total_sentences * train_ratio,
        "dev": total_sentences * dev_ratio,
        "test": total_sentences * test_ratio,
    }
    buckets: dict[str, list[Sentence]] = {"train": [], "dev": [], "test": []}

    # Sort larger groups first after random tie-breaking so bucket sizes stay close to target.
    for key in group_keys:
        group = groups[key]
        # Pick the split with the largest relative remaining deficit.
        chosen = max(
            ("train", "dev", "test"),
            key=lambda split: (
                (targets[split] - len(buckets[split])) / targets[split]
                if targets[split] > 0
                else float("-inf")
            ),
        )
        buckets[chosen].extend(group)

    return buckets["train"], buckets["dev"], buckets["test"]


def verify_no_leakage(
    train: Sequence[Sentence],
    dev: Sequence[Sentence],
    test: Sequence[Sentence],
) -> dict[str, Any]:
    """Check for document-ID overlap and exact text overlap across splits."""
    splits = {"train": train, "dev": dev, "test": test}
    doc_sets = {
        name: {s.group_key for s in sents if s.group_key}
        for name, sents in splits.items()
    }
    text_sets = {
        name: {" ".join(s.tokens) for s in sents}
        for name, sents in splits.items()
    }

    doc_overlap: dict[str, list[str]] = {}
    text_overlap: dict[str, int] = {}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        shared_docs = sorted(doc_sets[left] & doc_sets[right])
        if shared_docs:
            doc_overlap[key] = shared_docs
        shared_texts = text_sets[left] & text_sets[right]
        if shared_texts:
            text_overlap[key] = len(shared_texts)

    return {
        "has_doc_leakage": bool(doc_overlap),
        "has_text_leakage": bool(text_overlap),
        "doc_overlap": doc_overlap,
        "text_overlap": text_overlap,
    }


def align_labels_with_word_ids(
    word_tags: Sequence[str],
    word_ids: Sequence[int | None],
    *,
    label2id: dict[str, int] = LABEL2ID,
    label_all_tokens: bool = False,
) -> list[int]:
    """Align word-level BIO tags to tokenizer subword positions.

    Special tokens (``word_id is None``) always receive ``IGNORE_INDEX`` (-100).
    When ``label_all_tokens=False`` (recommended for NER), only the first subword
    of each word receives its BIO label ID; continuation subwords receive -100.
    When ``label_all_tokens=True``, continuation subwords of a ``B-X`` word are
    mapped to ``I-X`` so entity boundaries are not duplicated.
    """
    aligned: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            tag = word_tags[word_id]
            aligned.append(label2id[tag])
        else:
            if label_all_tokens:
                tag = word_tags[word_id]
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                aligned.append(label2id[tag])
            else:
                aligned.append(IGNORE_INDEX)
        previous_word_id = word_id

    return aligned


def decode_word_predictions(
    pred_ids: Sequence[int],
    word_ids: Sequence[int | None],
    *,
    num_words: int,
    id2label: dict[int, str] = ID2LABEL,
) -> tuple[list[str], int]:
    """Map subword-level predicted label IDs back to word-level BIO tags.

    Uses the prediction of the first subword of each word. Any trailing words
    dropped due to ``max_length`` truncation default to ``"O"``.

    Returns ``(word_tags, truncated_word_count)``.
    """
    word_tags: list[str] = ["O"] * num_words
    seen_words: set[int] = set()

    for pred_id, word_id in zip(pred_ids, word_ids):
        if word_id is None or word_id in seen_words or word_id >= num_words:
            continue
        word_tags[word_id] = id2label[int(pred_id)]
        seen_words.add(word_id)

    truncated = num_words - len(seen_words)
    return word_tags, truncated


def encode_sentences(
    sentences: Sequence[Sentence],
    tokenizer: Any,
    *,
    max_length: int = 256,
    label2id: dict[str, int] = LABEL2ID,
    label_all_tokens: bool = False,
    include_labels: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tokenize pre-segmented sentences and align BIO labels to subwords.

    Returns ``(encoded_examples, stats)`` where ``stats`` reports how many
    sentences and words were truncated by ``max_length``.
    """
    if not sentences:
        return [], {"sentences": 0, "truncated_sentences": 0, "truncated_words": 0}

    token_lists = [s.tokens for s in sentences]
    batch = tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    examples: list[dict[str, Any]] = []
    truncated_sentences = 0
    truncated_words = 0

    for index, sentence in enumerate(sentences):
        word_ids = batch.word_ids(batch_index=index)
        covered_words = {wid for wid in word_ids if wid is not None}
        dropped = len(sentence.tokens) - len(covered_words)
        if dropped > 0:
            truncated_sentences += 1
            truncated_words += dropped

        item: dict[str, Any] = {
            "input_ids": batch["input_ids"][index],
            "attention_mask": batch["attention_mask"][index],
            "word_ids": word_ids,
        }
        if "token_type_ids" in batch:
            item["token_type_ids"] = batch["token_type_ids"][index]
        if include_labels:
            item["labels"] = align_labels_with_word_ids(
                sentence.tags,
                word_ids,
                label2id=label2id,
                label_all_tokens=label_all_tokens,
            )
        examples.append(item)

    stats = {
        "sentences": len(sentences),
        "truncated_sentences": truncated_sentences,
        "truncated_words": truncated_words,
    }
    return examples, stats


def dataset_summary(sentences: Sequence[Sentence]) -> dict[str, Any]:
    """Compute basic corpus statistics (sentences, tokens, docs, label counts)."""
    from collections import Counter

    tag_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    docs = {s.group_key for s in sentences if s.group_key}
    total_tokens = 0

    for sentence in sentences:
        total_tokens += len(sentence.tokens)
        tag_counts.update(sentence.tags)
        prev = "O"
        for tag in sentence.tags:
            if tag.startswith("B-") or (
                tag.startswith("I-") and prev not in (f"B-{tag[2:]}", f"I-{tag[2:]}")
            ):
                entity_counts[tag[2:]] += 1
            prev = tag

    return {
        "sentences": len(sentences),
        "tokens": total_tokens,
        "documents": len(docs),
        "avg_tokens_per_sentence": round(total_tokens / len(sentences), 2) if sentences else 0.0,
        "entity_counts": {etype: entity_counts.get(etype, 0) for etype in ENTITY_TYPES},
        "total_entities": sum(entity_counts.values()),
        "tag_counts": dict(tag_counts),
    }


def make_synthetic_dataset(
    num_docs: int = 12,
    sentences_per_doc: int = 2,
    seed: int = 42,
) -> list[Sentence]:
    """Generate a tiny Khmer BIO dataset for smoke tests and unit tests."""
    templates: list[tuple[list[str], list[str]]] = [
        (
            ["លោក", "សុខ", "ដារ៉ា", "ធ្វើការ", "នៅ", "ធនាគារ", "ជាតិ", "ក្នុង", "ភ្នំពេញ", "។"],
            ["O", "B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
        ),
        (
            ["ក្រសួង", "សុខាភិបាល", "បាន", "ជួប", "លោកស្រី", "ចាន់", "ម៉ាលី", "នៅ", "សៀមរាប", "។"],
            ["B-ORG", "I-ORG", "O", "O", "O", "B-PER", "I-PER", "O", "B-LOC", "O"],
        ),
        (
            ["ពលរដ្ឋ", "នៅ", "ខេត្ត", "កំពត", "ស្នើ", "ទៅ", "រដ្ឋាភិបាល", "កម្ពុជា", "។"],
            ["O", "O", "B-LOC", "I-LOC", "O", "O", "B-ORG", "I-ORG", "O"],
        ),
        (
            ["អ្នកស្រី", "សុភា", "ទៅ", "សាកលវិទ្យាល័យ", "ភូមិន្ទ", "ភ្នំពេញ", "។"],
            ["O", "B-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O"],
        ),
    ]
    rng = random.Random(seed)
    output: list[Sentence] = []
    for doc_idx in range(1, num_docs + 1):
        doc_id = f"synthetic-doc-{doc_idx:03d}"
        for sent_idx in range(1, sentences_per_doc + 1):
            tokens, tags = rng.choice(templates)
            output.append(
                Sentence(
                    tokens=list(tokens),
                    tags=list(tags),
                    doc_id=doc_id,
                    sent_id=f"{doc_id}-{sent_idx:02d}",
                    meta={"source": "synthetic", "url": f"https://example.org/{doc_id}"},
                )
            )
    return output
