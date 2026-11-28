"""Stage 2b: Cluster-safe Khmer word segmentation and annotation batch export.

Reads sentence JSONL files written by ``cleaner/clean_text.py`` (typically
``data/processed/*_sentences.jsonl``), removes non-sentence boilerplate
(headings, date-only lines, photo/source captions), deduplicates repeated
sentences, segments each sentence into whole Khmer words without ever splitting
a Khmer Character Cluster (KCC), and exports annotation-ready files for
INCEpTION (one space-segmented sentence per line) alongside matching JSONL
provenance metadata.

If zero-width spaces (``\\u200b``) were preserved via ``clean_text.py --keep-zwsp``
(e.g., ODC text), they are used as word-boundary hints before Viterbi DP and
stripped from the final tokens.

Examples (run from the repository root):

    # Segment all data/processed/*_sentences.jsonl files
    python cleaner/segment_words.py

    # Segment specific files and export INCEpTION batches (smoke_5, pilot_60, pool + 6 annotators)
    python cleaner/segment_words.py \\
        --input data/processed/voa_khmer_sentences.jsonl \\
        --export-dir data/annotation_batches \\
        --pilot-size 60 --pool-size 2000 --annotators 6 --double-labeled 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner.clean_text import ZWSP, clean_text, khmer_chars, read_jsonl

PROCESSED_DIR = Path("data/processed")

# Khmer Unicode classes (using \uXXXX escapes for combining/invisible marks).
KHMER_COENG = "\u17d2"
KHMER_COMBINING_MARKS = frozenset(
    [
        *(chr(cp) for cp in range(0x17B6, 0x17D2)),  # dependent vowels + signs
        "\u17d3",  # bathamasat
        "\u17dd",  # atthacan
    ]
)
KHMER_CLUSTER_RE = re.compile(
    r"([\u1780-\u17a2\u17a5-\u17b3])"
    r"((?:\u17d2[\u1780-\u17b3]|[\u17b6-\u17d1\u17d3\u17dd])*)"
)

# Tokenization regex over cleaned text (handles Khmer runs, "។ល។", numerals,
# English words/initials/acronyms, and punctuation).
CHUNK_RE = re.compile(
    r"\u17d4\s*\u179b\s*\u17d4"  # Khmer "etc." (។ល។ / ។ ល។) as one token
    r"|[\u1780-\u17dd\u200c\u200d]+"  # Continuous Khmer letters/marks
    r"|[\u17e0-\u17e9]+(?:[.,][\u17e0-\u17e9]+)*"  # Khmer numerals
    r"|[0-9]+(?:[.,][0-9]+)*"  # Arabic numerals
    r"|[A-Za-z]\."  # Latin initial (e.g. J. in Samuel J. Paparo)
    r"|[A-Za-z]+(?:[-'.][A-Za-z0-9]+)*"  # English words / acronyms (USAID, KF-16)
    r"|[^\s]"  # Standalone punctuation or symbol
)

_SENTENCE_END_RE = re.compile(r"[\u17d4\u17d5\u17da?!]")
_DATE_ONLY_RE = re.compile(
    r"^(?:(?:បាន)?ធ្វើបច្ចុប្បន្នភាព|ចេញផ្សាយ(?:លើកដំបូង)?|កែសម្រួល|ថ្ងៃទី)\s*[:៖]?"
    r"[\s0-9\u17e0-\u17e9ក-៿,/.-]{0,90}$"
)
_CAPTION_PREFIXES = (
    "ប្រភព៖",
    "ប្រភព:",
    "រូបភាព៖",
    "រូបភាព:",
    "រូបថត៖",
    "រូបថត:",
    "source:",
    "photo:",
    "image:",
    "credit:",
)

# High-frequency function/title prefixes that should not stay glued to entity nouns.
NON_ENTITY_GLUE_PREFIXES: tuple[str, ...] = (
    "ជាមួយ",
    "របស់",
    "សម្រាប់",
    "ដើម្បី",
    "បន្ទាប់ពី",
    "កាលពី",
    "ចំពោះ",
    "អំពី",
    "ដោយ",
    "តាម",
    "ក្នុង",
    "នៅ",
    "លើ",
    "ក្រោម",
    "ពី",
    "នៃ",
    "និង",
    "ឬ",
    "ឱ្យ",
    "ឲ្យ",
    "បាន",
    "នឹង",
    "ដែល",
    "ថា",
    "លោក",
    "លោកស្រី",
    "អ្នកស្រី",
    "កញ្ញា",
)

# Core Khmer news, policy, encyclopedic, and NER-boundary vocabulary (offline).
CORE_KHMER_LEXICON: dict[str, int] = {
    # Pronouns, verbs, prepositions, conjunctions
    "ខ្ញុំ": 500, "គាត់": 500, "ពួកគេ": 400, "យើង": 400, "ខ្លួន": 400, "គេ": 400,
    "បាន": 900, "ជួប": 300, "ស្គាល់": 300, "ទៅ": 800, "មក": 700, "នៅ": 900,
    "ក្នុង": 800, "លើ": 600, "ក្រោម": 300, "ពី": 800, "ដល់": 500, "ដោយ": 700,
    "តាម": 600, "ដើម្បី": 600, "សម្រាប់": 500, "ជាមួយ": 500, "អំពី": 500,
    "និង": 900, "ឬ": 500, "ប៉ុន្តែ": 500, "ហើយ": 800, "ថា": 900, "ដែល": 900,
    "ជា": 900, "មាន": 900, "មិន": 800, "ត្រូវ": 700, "អាច": 600, "នឹង": 800,
    "នេះ": 800, "នោះ": 600, "ទាំង": 500, "មួយ": 700, "ពីរ": 400, "បី": 300,
    "ការ": 900, "អ្នក": 800, "របស់": 800, "នៃ": 700, "ឱ្យ": 700, "ឲ្យ": 500,
    "ធ្វើ": 700, "ប្រាប់": 500, "ថ្លែង": 500, "និយាយ": 500, "បញ្ជាក់": 500,
    "លើកឡើង": 400, "បន្ត": 500, "បន្ថែម": 500, "ស្នើ": 400, "ជំរុញ": 400,
    "ហៅ": 300, "កាត់": 300, "ហៅកាត់": 350, "ប្រឆាំង": 400, "គាំទ្រ": 400,
    "ប្រឈម": 400, "ទទួល": 500, "ផ្តល់": 500, "ផ្ដល់": 400, "ចាប់": 400,
    "ដោះលែង": 400, "ស៊ើបអង្កេត": 400, "កាត់ទោស": 350, "បដិសេធ": 400,
    "ធ្វើការ": 450, "ប្រើប្រាស់": 450, "អភិវឌ្ឍន៍": 450, "គ្រប់គ្រង": 450,
    # Titles & honorifics (Rulebook Case 1 & 2)
    "លោក": 900, "លោកស្រី": 600, "កញ្ញា": 400, "អ្នកស្រី": 400, "អ្នកនាង": 300,
    "សម្តេច": 500, "សម្ដេច": 400, "ព្រះមហាក្សត្រ": 400, "ឯកឧត្តម": 400,
    "បណ្ឌិត": 400, "ប្រធាន": 600, "ប្រធានាធិបតី": 600, "នាយករដ្ឋមន្ត្រី": 600,
    "រដ្ឋមន្ត្រី": 500, "មន្ត្រី": 600, "មន្រ្តី": 500, "អ្នកនាំពាក្យ": 500,
    "សកម្មជន": 450, "មេធាវី": 400, "ចៅក្រម": 400, "ព្រះរាជអាជ្ញា": 400,
    "ពលរដ្ឋ": 600, "ប្រជាពលរដ្ឋ": 500, "កម្មករ": 500, "ពលករ": 450, "អ្នកសារព័ត៌មាន": 450,
    # Administrative place prefixes & Cambodian / regional locations (Rulebook Case 5)
    "រាជធានី": 500, "ខេត្ត": 700, "ក្រុង": 600, "ខណ្ឌ": 500, "ស្រុក": 500,
    "ឃុំ": 400, "សង្កាត់": 450, "ភូមិ": 450, "តំបន់": 500, "ប្រទេស": 700,
    "កម្ពុជា": 900, "ភ្នំពេញ": 800, "កោះកុង": 400, "ឫស្សីកែវ": 400, "ជ្រោយចង្វារ": 400,
    "ទួលគោក": 350, "សែនសុខ": 350, "សៀមរាប": 450, "បាត់ដំបង": 400, "កំពត": 400,
    "កំពង់ចាម": 400, "កំពង់ធំ": 400, "កណ្តាល": 400, "កណ្ដាល": 350, "ព្រះសីហនុ": 450,
    "បន្ទាយមានជ័យ": 400, "ព្រៃវែង": 400, "ស្វាយរៀង": 400, "តាកែវ": 400, "ពោធិ៍សាត់": 400,
    "ស្ទឹងត្រែង": 350, "រតនគិរី": 350, "មណ្ឌលគិរី": 350, "ក្រចេះ": 350, "ព្រះវិហារ": 350,
    "ប៉ៃលិន": 350, "កែប": 350, "ឧត្តរមានជ័យ": 350, "ត្បូងឃ្មុំ": 350, "កំពង់ស្ពឺ": 400,
    "កំពង់ឆ្នាំង": 400, "សហរដ្ឋអាមេរិក": 700, "អាមេរិក": 700, "ចិន": 600, "ថៃ": 600,
    "វៀតណាម": 600, "ឡាវ": 500, "ម៉ាឡេស៊ី": 450, "សិង្ហបុរី": 400, "ជប៉ុន": 500,
    "កូរ៉េខាងត្បូង": 500, "កូរ៉េខាងជើង": 450, "រុស្ស៊ី": 550, "អ៊ុយក្រែន": 550,
    "អ៊ីស្រាអែល": 500, "ហ្កាហ្សា": 450, "បារាំង": 450, "អង់គ្លេស": 450, "អឺរ៉ុប": 450,
    "អាស៊ាន": 500, "អាស៊ី": 500, "កាណាដា": 400, "មេគង្គ": 400, "ទន្លេសាប": 400,
    # Organizations & institutions (Rulebook Case 3 & 4)
    "រដ្ឋាភិបាល": 700, "ក្រសួង": 700, "តុលាការ": 650, "សាលាដំបូង": 500,
    "សាលាឧទ្ធរណ៍": 450, "តុលាការកំពូល": 500, "ក្រសួងមហាផ្ទៃ": 500,
    "ក្រសួងការបរទេស": 500, "ក្រសួងយុត្តិធម៌": 500, "ក្រសួងការពារជាតិ": 450,
    "ក្រសួងសេដ្ឋកិច្ច": 450, "ក្រសួងអប់រំ": 450, "ក្រសួងសុខាភិបាល": 450,
    "ក្រសួងបរិស្ថាន": 450, "ក្រសួងការងារ": 450, "ក្រសួងកសិកម្ម": 450,
    "គណបក្ស": 600, "សមាគម": 550, "អង្គការ": 700, "សហប្រជាជាតិ": 500,
    "ក្រុមហ៊ុន": 650, "ធនាគារ": 550, "សាកលវិទ្យាល័យ": 450, "មន្ទីរពេទ្យ": 450,
    "រដ្ឋសភា": 500, "ព្រឹទ្ធសភា": 450, "វីអូអេ": 700, "សេតវិមាន": 450,
    "អូតង់": 400, "សហភាពអឺរ៉ុប": 450, "គណៈកម្មាធិការ": 450,
    "ប្រជាធិបតេយ្យ": 500, "ឯករាជ្យ": 450, "សេដ្ឋកិច្ច": 600, "ក្រៅប្រព័ន្ធ": 400,
    "សិទ្ធិមនុស្ស": 600, "នយោបាយ": 600, "សង្គម": 550, "សង្គមស៊ីវិល": 450,
    "សារព័ត៌មាន": 550, "ប្រព័ន្ធផ្សព្វផ្សាយ": 450, "ទីភ្នាក់ងារ": 500,
    "បរិស្ថាន": 500, "អប់រំ": 500, "សុខភាព": 500, "ច្បាប់": 550, "ពន្ធនាគារ": 400,
    "កសិកម្ម": 450, "ព្រៃឈើ": 450, "ជីវៈចម្រុះ": 400, "ធនធាន": 450,
    "ឆ្នាំ": 700, "ខែ": 600, "ថ្ងៃ": 650, "សប្តាហ៍": 450, "ដុល្លារ": 550,
}


def split_khmer_clusters(khmer_run: str) -> list[str]:
    """Split a continuous normalized Khmer string into atomic orthographic clusters."""
    clusters: list[str] = []
    pos = 0
    length = len(khmer_run)

    for match in KHMER_CLUSTER_RE.finditer(khmer_run):
        if match.start() > pos:
            stray = khmer_run[pos : match.start()]
            if clusters and all(
                ch in KHMER_COMBINING_MARKS or ch in (KHMER_COENG, "\u200c", "\u200d")
                for ch in stray
            ):
                clusters[-1] += stray
            else:
                clusters.append(stray)
        clusters.append(match.group(0))
        pos = match.end()

    if pos < length:
        remainder = khmer_run[pos:]
        if clusters and all(
            ch in KHMER_COMBINING_MARKS or ch in (KHMER_COENG, "\u200c", "\u200d")
            for ch in remainder
        ):
            clusters[-1] += remainder
        else:
            clusters.append(remainder)

    return clusters


def has_broken_khmer_token(tokens: Sequence[str]) -> bool:
    """Return True if any token starts with a Khmer combining mark or coeng."""
    for tok in tokens:
        if tok and (tok[0] in KHMER_COMBINING_MARKS or tok[0] == KHMER_COENG):
            return True
    return False


def is_boilerplate_sentence(
    text: str,
    *,
    drop_headings: bool = True,
    max_heading_chars: int = 45,
) -> str:
    """Return a skip reason if ``text`` looks like a caption, date line, or heading."""
    stripped = text.replace(ZWSP, "").strip()
    if not stripped:
        return "empty"
    lowered = stripped.casefold()
    if any(lowered.startswith(prefix) for prefix in _CAPTION_PREFIXES):
        return "caption-or-source"
    if _DATE_ONLY_RE.match(stripped):
        return "date-line"
    if drop_headings and len(stripped) < max_heading_chars and not _SENTENCE_END_RE.search(stripped):
        return "heading-like"
    return ""


def build_lexicon_from_texts(
    texts: Iterable[str],
    *,
    min_freq: int = 3,
    max_word_chars: int = 16,
) -> dict[str, int]:
    """Expand ``CORE_KHMER_LEXICON`` with frequent normalized words from corpus boundaries."""
    counts: Counter[str] = Counter(CORE_KHMER_LEXICON)
    raw_counts: Counter[str] = Counter()

    for raw_text in texts:
        for piece in re.split(r"[\u200b\s\u17d4\u17d5\u17da?!«»\"'()\[\],.:;\u17d6\u17d7]+", raw_text):
            if not piece:
                continue
            norm = clean_text(piece, keep_zwsp=False)
            if re.fullmatch(r"[\u1780-\u17dd]+", norm) and 2 <= len(norm) <= max_word_chars:
                raw_counts[norm] += 1

    known_roots = set(CORE_KHMER_LEXICON) | {
        word for word, freq in raw_counts.items() if freq >= 5 and len(word) <= 12
    }
    for word, freq in raw_counts.items():
        if freq < min_freq:
            continue
        if word in CORE_KHMER_LEXICON:
            counts[word] = max(counts[word], freq)
            continue
        glued = any(
            word.startswith(prefix)
            and len(word) > len(prefix)
            and word[len(prefix) :] in known_roots
            for prefix in NON_ENTITY_GLUE_PREFIXES
        )
        if not glued:
            counts[word] = max(counts.get(word, 0), freq)

    return dict(counts)


class KhmerWordSegmenter:
    """KCC-safe Khmer word segmenter using Unigram Viterbi DP.

    If input text contains zero-width spaces (``\\u200b``), they are honored as
    word-boundary hints before running Viterbi DP on each Khmer segment, and no
    ``\\u200b`` characters remain in the returned tokens.
    """

    def __init__(
        self,
        lexicon: dict[str, int] | None = None,
        *,
        max_clusters_per_word: int = 8,
    ) -> None:
        self.lexicon: dict[str, int] = dict(lexicon or CORE_KHMER_LEXICON)
        self.max_clusters_per_word = max_clusters_per_word
        self.total_freq = float(sum(self.lexicon.values()) + 1000)
        self._log_probs: dict[str, float] = {
            word: math.log((freq + 1.0) / self.total_freq)
            for word, freq in self.lexicon.items()
        }
        self._unk_cluster_penalty = math.log(0.15 / self.total_freq)

    def segment_khmer_run(self, khmer_run: str) -> list[str]:
        """Segment one continuous Khmer script run into whole words."""
        clusters = split_khmer_clusters(khmer_run)
        n = len(clusters)
        if n <= 1:
            return clusters

        dp: list[tuple[float, int]] = [(-1e18, -1)] * (n + 1)
        dp[0] = (0.0, -1)

        for i in range(n):
            base_score = dp[i][0]
            if base_score <= -1e17:
                continue
            max_j = min(n, i + self.max_clusters_per_word)
            candidate = ""
            for j in range(i + 1, max_j + 1):
                candidate += clusters[j - 1]
                if candidate in self._log_probs:
                    length_bonus = 0.25 * (j - i - 1)
                    score = base_score + self._log_probs[candidate] + length_bonus
                    if score > dp[j][0]:
                        dp[j] = (score, i)
                elif j == i + 1:
                    score = base_score + self._unk_cluster_penalty
                    if score > dp[j][0]:
                        dp[j] = (score, i)

        words: list[str] = []
        cursor = n
        while cursor > 0:
            prev = dp[cursor][1]
            if prev < 0:
                words.append(clusters[cursor - 1])
                cursor -= 1
            else:
                words.append("".join(clusters[prev:cursor]))
                cursor = prev
        words.reverse()
        return words

    def segment(self, text: str, *, use_zwsp_hints: bool = True) -> list[str]:
        """Normalize and segment a Khmer or mixed-script sentence into tokens."""
        cleaned = clean_text(text, keep_zwsp=use_zwsp_hints)
        if not cleaned:
            return []

        parts = cleaned.split(ZWSP) if use_zwsp_hints and ZWSP in cleaned else [cleaned]
        tokens: list[str] = []
        for part in parts:
            for match in CHUNK_RE.finditer(part):
                chunk = match.group(0)
                if re.fullmatch(r"\u17d4\s*\u179b\s*\u17d4", chunk):
                    tokens.append("\u17d4\u179b\u17d4")
                elif "\u1780" <= chunk[0] <= "\u17dd":
                    tokens.extend(self.segment_khmer_run(chunk))
                else:
                    tokens.append(chunk)
        return tokens


def segment_records(
    records: Sequence[dict[str, Any]],
    segmenter: KhmerWordSegmenter,
    *,
    min_khmer: int = 10,
    max_chars: int = 600,
    drop_boilerplate: bool = True,
    drop_headings: bool = True,
    deduplicate: bool = True,
    use_zwsp_hints: bool = True,
) -> tuple[list[dict[str, Any]], Counter]:
    """Filter, deduplicate, and word-segment sentence JSONL records."""
    counts: Counter = Counter()
    seen_hashes: set[str] = set()
    output: list[dict[str, Any]] = []

    for record in records:
        counts["input_sentences"] += 1
        raw_text = str(record.get("text") or "")
        normalized_text = clean_text(raw_text, keep_zwsp=False)
        if khmer_chars(normalized_text) < min_khmer:
            counts["dropped_short"] += 1
            continue
        if max_chars > 0 and len(normalized_text) > max_chars:
            counts["dropped_too_long"] += 1
            continue
        if drop_boilerplate:
            reason = is_boilerplate_sentence(normalized_text, drop_headings=drop_headings)
            if reason:
                counts[f"dropped_{reason}"] += 1
                continue

        if deduplicate:
            digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                counts["dropped_duplicate"] += 1
                continue
            seen_hashes.add(digest)

        tokens = segmenter.segment(raw_text, use_zwsp_hints=use_zwsp_hints)
        if not tokens:
            counts["dropped_empty"] += 1
            continue
        if has_broken_khmer_token(tokens):
            counts["broken_khmer_cluster_sentences"] += 1

        counts["kept_sentences"] += 1
        counts["total_tokens"] += len(tokens)
        out = dict(record)
        out["text"] = normalized_text
        out["tokens"] = tokens
        out["token_count"] = len(tokens)
        out["segmented_text"] = " ".join(tokens)
        out["changed"] = True
        output.append(out)

    return output, counts


def _write_batch(rows: Sequence[dict[str, Any]], stem_path: Path) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = stem_path.with_suffix(".txt")
    jsonl_path = stem_path.with_suffix(".jsonl")
    with (
        txt_path.open("w", encoding="utf-8", newline="\n") as txt_f,
        jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_f,
    ):
        for row in rows:
            txt_f.write(row["segmented_text"] + "\n")
            jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_annotation_batches(
    rows: Sequence[dict[str, Any]],
    export_dir: Path,
    *,
    smoke_size: int = 5,
    pilot_size: int = 60,
    pool_size: int = 2000,
    annotators: int = 0,
    double_labeled: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Export deterministic INCEpTION-ready .txt + matching .jsonl provenance batches."""
    export_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    # Pick pilot sentences from distinct documents first for broad coverage.
    pilot: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    used_indices: set[int] = set()
    for idx in indices:
        doc_key = str(rows[idx].get("doc_id") or rows[idx].get("url") or idx)
        if doc_key not in seen_docs:
            pilot.append(rows[idx])
            seen_docs.add(doc_key)
            used_indices.add(idx)
        if len(pilot) >= min(pilot_size, len(rows)):
            break
    if len(pilot) < min(pilot_size, len(rows)):
        for idx in indices:
            if idx not in used_indices:
                pilot.append(rows[idx])
                used_indices.add(idx)
            if len(pilot) >= min(pilot_size, len(rows)):
                break

    smoke = pilot[: min(smoke_size, len(pilot))]
    remaining = [rows[idx] for idx in indices if idx not in used_indices]
    pool = remaining[: min(pool_size, len(remaining))]

    if smoke:
        _write_batch(smoke, export_dir / f"smoke_{len(smoke)}")
    if pilot:
        _write_batch(pilot, export_dir / f"pilot_{len(pilot)}")
    if pool:
        _write_batch(pool, export_dir / f"annotation_pool_{len(pool)}")

    annotator_summary: dict[str, int] = {}
    if annotators > 0 and pool:
        overlap_count = min(double_labeled, len(pool)) if annotators >= 2 else 0
        overlap_rows = pool[:overlap_count]
        unique_rows = pool[overlap_count:]

        if overlap_rows:
            _write_batch(overlap_rows, export_dir / f"iaa_double_{len(overlap_rows)}")

        # Assign each overlap sentence to two rotating annotators: (i % k) and ((i + 1) % k).
        per_annotator: list[list[dict[str, Any]]] = [[] for _ in range(annotators)]
        for i, row in enumerate(overlap_rows):
            per_annotator[i % annotators].append(row)
            per_annotator[(i + 1) % annotators].append(row)
        for i, row in enumerate(unique_rows):
            per_annotator[i % annotators].append(row)

        for idx, batch in enumerate(per_annotator, 1):
            name = f"annotator_{idx}"
            _write_batch(batch, export_dir / name)
            annotator_summary[name] = len(batch)

    manifest = {
        "seed": seed,
        "total_available": len(rows),
        "smoke_count": len(smoke),
        "pilot_count": len(pilot),
        "pool_count": len(pool),
        "double_labeled_count": min(double_labeled, len(pool)) if annotators >= 2 else 0,
        "annotator_batches": annotator_summary,
    }
    (export_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def find_sentence_inputs(processed_dir: Path) -> list[Path]:
    return sorted(processed_dir.glob("*_sentences.jsonl"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment cleaned Khmer sentences into KCC-safe words and export annotation batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        help="sentence JSONL files (default: data/processed/*_sentences.jsonl)",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--min-khmer", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--keep-boilerplate", action="store_true")
    parser.add_argument("--keep-headings", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--ignore-zwsp-hints", action="store_true")
    parser.add_argument("--export-dir", type=Path, help="export smoke/pilot/pool INCEpTION batches here")
    parser.add_argument("--smoke-size", type=int, default=5)
    parser.add_argument("--pilot-size", type=int, default=60)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--annotators", type=int, default=0, help="optional number of annotator batches (e.g. 6)")
    parser.add_argument("--double-labeled", type=int, default=200, help="shared IAA sentences when --annotators >= 2")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or find_sentence_inputs(PROCESSED_DIR)
    if not inputs:
        print(
            f"No *_sentences.jsonl files found in {PROCESSED_DIR}/. Run cleaner/clean_text.py first.",
            file=sys.stderr,
        )
        return 1

    loaded_by_file: list[tuple[Path, list[dict[str, Any]]]] = []
    all_texts: list[str] = []
    for source in inputs:
        if not source.is_file():
            print(f"{source}: not found", file=sys.stderr)
            return 1
        rows = list(read_jsonl(source))
        loaded_by_file.append((source, rows))
        all_texts.extend(str(r.get("text") or "") for r in rows)

    lexicon = build_lexicon_from_texts(all_texts)
    segmenter = KhmerWordSegmenter(lexicon=lexicon)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_segmented: list[dict[str, Any]] = []

    for source, rows in loaded_by_file:
        segmented, counts = segment_records(
            rows,
            segmenter,
            min_khmer=args.min_khmer,
            max_chars=args.max_chars,
            drop_boilerplate=not args.keep_boilerplate,
            drop_headings=not args.keep_headings,
            deduplicate=not args.keep_duplicates,
            use_zwsp_hints=not args.ignore_zwsp_hints,
        )
        stem = source.stem.removesuffix("_sentences")
        target = args.output_dir / f"{stem}_segmented.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as out_f:
            for row in segmented:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        avg_tok = counts["total_tokens"] / max(counts["kept_sentences"], 1)
        print(
            f"{source} -> {target}\n"
            f"  {counts['input_sentences']} -> {counts['kept_sentences']} sentences "
            f"({counts['total_tokens']} tokens, avg {avg_tok:.1f} tok/sent, "
            f"{counts['broken_khmer_cluster_sentences']} broken clusters)"
        )
        combined_segmented.extend(segmented)

    if args.export_dir is not None and combined_segmented:
        manifest = export_annotation_batches(
            combined_segmented,
            args.export_dir,
            smoke_size=args.smoke_size,
            pilot_size=args.pilot_size,
            pool_size=args.pool_size,
            annotators=args.annotators,
            double_labeled=args.double_labeled,
            seed=args.seed,
        )
        print(
            f"Exported annotation batches to {args.export_dir}/ "
            f"(smoke={manifest['smoke_count']}, pilot={manifest['pilot_count']}, "
            f"pool={manifest['pool_count']}, double_labeled={manifest['double_labeled_count']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
