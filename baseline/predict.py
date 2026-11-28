"""Run NER inference on space-segmented Khmer text, CoNLL, or JSONL files.

In addition to BIO predictions and extracted entity spans, JSONL output
includes ``least_confidence`` (``1 - min_word_prob``) and ``mean_entropy``
per sentence so the same script supports uncertainty-based Active Learning.

Examples (run from the repository root):

    python baseline/predict.py \\
        --model-dir outputs/xlmr_baseline/best_model \\
        --input data/cleaned/unlabeled.txt \\
        --output outputs/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.data import (
    ID2LABEL,
    Sentence,
    load_dataset_file,
    repair_bio_tags,
    write_conll,
)
from baseline.metrics import extract_spans


def load_unlabeled_input(path: Path) -> list[Sentence]:
    """Load space-segmented ``.txt``, ``.jsonl``, or ``.conll`` sentences."""
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".conll", ".bio"):
        return load_dataset_file(path, require_tags=False)

    sentences: list[Sentence] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            tokens = raw_line.strip().split()
            if not tokens:
                continue
            sentences.append(
                Sentence(
                    tokens=tokens,
                    tags=["O"] * len(tokens),
                    sent_id=f"{path.stem}-{line_number:05d}",
                )
            )
    return sentences


def decode_word_probs(
    probs_matrix: Sequence[Sequence[float]],
    word_ids: Sequence[int | None],
    *,
    num_words: int,
    id2label: dict[int, str] = ID2LABEL,
) -> tuple[list[str], list[float], list[float]]:
    """Decode first-subword probabilities into word tags, confidences, and entropies."""
    tags: list[str] = ["O"] * num_words
    confidences: list[float] = [1.0] * num_words
    entropies: list[float] = [0.0] * num_words
    seen: set[int] = set()

    for prob_vec, word_id in zip(probs_matrix, word_ids):
        if word_id is None or word_id in seen or word_id >= num_words:
            continue
        best_idx = max(range(len(prob_vec)), key=lambda i: prob_vec[i])
        best_prob = float(prob_vec[best_idx])
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in prob_vec if p > 0.0)
        tags[word_id] = id2label[best_idx]
        confidences[word_id] = best_prob
        entropies[word_id] = entropy
        seen.add(word_id)

    return tags, confidences, entropies


def predict_sentences(
    model_dir: Path,
    sentences: Sequence[Sentence],
    *,
    max_length: int = 256,
    batch_size: int = 32,
    repair_bio: bool = True,
) -> list[dict[str, Any]]:
    """Predict BIO tags, entity spans, and uncertainty scores for each sentence."""
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

    results: list[dict[str, Any]] = []
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
            probs_batch = torch.softmax(logits, dim=-1).cpu().tolist()

        for index, (probs_matrix, sentence) in enumerate(zip(probs_batch, chunk)):
            word_ids = encoded.word_ids(batch_index=index)
            tags, confidences, entropies = decode_word_probs(
                probs_matrix,
                word_ids,
                num_words=len(sentence.tokens),
                id2label=id2label,
            )
            if repair_bio:
                tags = repair_bio_tags(tags)
            spans = extract_spans(tags)
            entities = [
                {
                    "type": span.entity_type,
                    "start": span.start,
                    "end": span.end,
                    "text": " ".join(sentence.tokens[span.start : span.end]),
                    "confidence": round(
                        sum(confidences[span.start : span.end]) / (span.end - span.start),
                        4,
                    ),
                }
                for span in spans
            ]
            least_confidence = round(1.0 - min(confidences, default=1.0), 6)
            mean_entropy = round(sum(entropies) / len(entropies), 6) if entropies else 0.0
            results.append(
                {
                    "doc_id": sentence.doc_id,
                    "sent_id": sentence.sent_id,
                    "tokens": sentence.tokens,
                    "tags": tags,
                    "entities": entities,
                    "least_confidence": least_confidence,
                    "mean_entropy": mean_entropy,
                }
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tag space-segmented Khmer text with a trained XLM-R NER checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help=".txt (space-segmented), .conll, or .jsonl")
    parser.add_argument("--output", type=Path, required=True, help=".jsonl or .conll output path")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-repair-bio", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sentences = load_unlabeled_input(args.input)
    if not sentences:
        raise SystemExit(f"No sentences found in {args.input}")

    predictions = predict_sentences(
        args.model_dir,
        sentences,
        max_length=args.max_length,
        batch_size=args.batch_size,
        repair_bio=not args.no_repair_bio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in (".conll", ".bio"):
        tagged = [
            Sentence(
                tokens=item["tokens"],
                tags=item["tags"],
                doc_id=item["doc_id"],
                sent_id=item["sent_id"],
            )
            for item in predictions
        ]
        write_conll(tagged, args.output)
    else:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Tagged {len(predictions)} sentence(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
