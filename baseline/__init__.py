"""XLM-RoBERTa baseline pipeline for Khmer NER (PER, LOC, ORG).

Reads standard CoNLL-2002 BIO-tagged datasets (train/dev/test), aligns word
labels to SentencePiece subwords, trains and evaluates a token-classification
model (default ``xlm-roberta-base``), and exports metrics, charts, and a
UTF-8 HTML error report.
"""

from baseline.data import ENTITY_TYPES, ID2LABEL, LABEL2ID, LABELS

__all__ = ["ENTITY_TYPES", "ID2LABEL", "LABEL2ID", "LABELS"]
