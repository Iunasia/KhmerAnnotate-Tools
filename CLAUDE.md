# KhmerAnnotate-Tools

Tools for building a Khmer NER dataset (labels `PER`, `LOC`, `ORG`).
Pipeline: `collector/` -> `cleaner/` -> annotate -> `checker/` -> `baseline/`.
See `README.md` for usage of each script.

## How annotation works (decided by the team)

- Articles are split into sentences; annotation is done sentence by sentence.
- Within a sentence, labels are applied word by word. Khmer has no spaces
  between words, so a word-segmentation step is required (not built yet).

## Commands

- Python: use the venv directly, `.venv\Scripts\python.exe` (Python 3.14).
- Tests: `.venv\Scripts\python.exe -m unittest discover -s tests -t .`
- Clean all raw data: `.venv\Scripts\python.exe cleaner/clean_text.py`
- Set `PYTHONIOENCODING=utf-8` when a script prints Khmer.

## Conventions

- `data/` is never committed (publisher licences). Raw data is shared through
  the private Hugging Face dataset `KhmerAnnotate/khmer-ner-raw`.
- RFA is not a source: its terms forbid automated collection. Do not re-add it.
- In source code, write invisible and combining-only characters as `\uXXXX`
  escapes, never as literal characters.
- Terminals cannot render Khmer correctly. Put Khmer output that a person
  must read into a file or HTML page, and show code points in the terminal.
- Write the README in plain, simple English.

## Current state (update this section after each piece of work)

Last updated: 2026-11-27

Done:
- Collectors for VOA, Wikipedia (with `--sample`/`--seed` and filters) and ODC.
- `cleaner/clean_text.py`: removes hidden characters, normalises spaces,
  fixes Khmer typing order and vowel encoding, removes repeated marks, stores
  subscript ដ/ត one way (`្ត`, except `្ដ` after `ណ`), splits sentences at
  `។ ៕ ៚ ? !`, drops sentences with fewer than 10 Khmer characters.
- Tests: `tests/test_collectors.py`, `tests/test_cleaner.py` (27 tests pass).
- Cleaner run on all three sources (output in `data/processed/`):
  ODC 238 articles -> 10,351 sentences; VOA 200 -> 3,055; Wikipedia 50 -> 1,968.
  Total 488 articles -> 15,374 sentences.
- `huggingface_hub` installed in the venv; VOA and Wikipedia raw files
  downloaded from the Hugging Face dataset into `data/raw/`.
- ODC blog inspected (5 posts): report in
  `data/processed/odc_blog_inspection.html`. Posts live at `/km/<slug>/`,
  not under `/km/blog/`; about 335 posts.

Open:
- ODC blog collector (`--include-blog`): proposal for skip rules is waiting
  for the team's OK; no code written yet.
- The Hugging Face dataset holds VOA and Wikipedia only; ODC raw data is not
  uploaded there.
- Wikipedia: 236 short sentences dropped by the cleaner (not yet looked at).
- The subscript ដ/ត rule was written from memory of SIL's `khnormal` and is
  unverified; it rewrites ODC's usual `ណ្ត` spelling (1,401 changes). The team
  must decide whether to keep it.
- Word segmentation for word-by-word annotation (candidate: `khmer-nltk`;
  zero-width spaces in ODC text are word boundaries and are currently removed
  unless `--keep-zwsp` is used).
- Output still contains headings, date lines and source captions (about 9%),
  46 duplicate sentences and a few very long sentences.
- The spaced `។ ល។` variant is split wrongly.
