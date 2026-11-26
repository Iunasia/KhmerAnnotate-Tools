# KhmerAnnotate-Tools

Tools for building a Khmer named-entity recognition (NER) dataset with the
labels `PER` (person), `LOC` (location), and `ORG` (organisation).

The planned pipeline is:

```text
collector/ -> cleaner/ -> annotate -> checker/ -> baseline/
raw text      Unicode      label       QA and     XLM-R NER
              cleanup      entities    agreement  model
```

## Project status

| Folder | Purpose | Status |
|---|---|---|
| `collector/` | Collect Khmer text from VOA and Wikipedia | Ready (250 raw articles collected) |
| `cleaner/` | Clean hidden Unicode characters and prepare sentences | Planned |
| `guidelines/` | Khmer annotation rulebook | Planned |
| `checker/` | Flag annotation mistakes and measure agreement | Planned |
| `baseline/` | Baseline XLM-RoBERTa model | Planned |
| `data/` | Local raw and processed data (not committed) | Shared on Hugging Face |

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optionally set a contact address used in collector user agents:

```text
KHMERANNOTATE_CONTACT_EMAIL=your-team@example.com
```

## Get the shared raw data (team members)

The raw articles live in a private Hugging Face dataset,
[`KhmerAnnotate/khmer-ner-raw`](https://huggingface.co/datasets/KhmerAnnotate/khmer-ner-raw).
Ask the project lead for an invite to the `KhmerAnnotate` organisation, then:

```powershell
pip install huggingface_hub
hf auth login        # choose "Log in with your browser" and grant access to KhmerAnnotate
hf download KhmerAnnotate/khmer-ner-raw --repo-type dataset --local-dir data/raw
```

This puts the files where the scripts expect them:

| File | Articles |
|---|---|
| `data/raw/voa_khmer/voa_khmer.jsonl` | 200 VOA Khmer news articles (Jan–Mar 2025) |
| `data/raw/wikipedia_km/wikipedia_km.jsonl` | 50 Khmer Wikipedia articles (random sample, seed 42) |

To upload new or updated files (needs write access):

```powershell
hf upload KhmerAnnotate/khmer-ner-raw data/raw . --repo-type dataset --include "*.jsonl"
```

**MSYS2 Python note:** `pip install huggingface_hub` fails there because its
optional `hf-xet` dependency cannot be built. Install it without that part:

```powershell
pip install --no-deps huggingface_hub
pip install click filelock fsspec httpx2 packaging pyyaml tqdm typing-extensions
```

On MSYS2 the `hf` command is in `.venv\bin\hf.exe`.

## Collect VOA Khmer news

The VOA collector checks `robots.txt`, waits between requests, skips likely
AP/Reuters/AFP material, and appends only URLs that are not already saved.

VOA Khmer stopped publishing on 15 March 2025, so category pages only show a
limited set of old articles. Use `--source sitemap` to collect more.

```bash
# Articles from category pages
python collector/voa_khmer_scraper.py --max 50

# A larger collection using sitemaps (recommended)
python collector/voa_khmer_scraper.py --source sitemap --max 500 --delay 3

# Replace the existing JSONL instead of resuming it
python collector/voa_khmer_scraper.py --overwrite --max 50
```

Output: `data/raw/voa_khmer/voa_khmer.jsonl`

VOA-produced text is public domain, but third-party wire material is not. The
automatic filter is a safeguard, not a legal guarantee; review records before
publishing a dataset.

## Collect Khmer Wikipedia text

This collector downloads the official compressed Khmer Wikipedia dump and
streams it, so the full dump is never loaded into memory.

```bash
# Download the dump and export 50 random articles (same 50 for the same seed)
python collector/wikipedia_khmer_scraper.py --max 50 --sample --seed 42

# Export the first 1,000 articles in dump order
python collector/wikipedia_khmer_scraper.py --max 1000

# Process a dump that is already on disk
python collector/wikipedia_khmer_scraper.py --dump path/to/kmwiki.xml.bz2 --max 1000

# Download without extracting articles yet
python collector/wikipedia_khmer_scraper.py --download-only
```

By default, articles shorter than 1,500 characters after cleaning, or less
than 50% Khmer script, are skipped (`--min-chars`, `--min-khmer`).

Output: `data/raw/wikipedia_km/wikipedia_km.jsonl`

Wikipedia is encyclopaedic text, not news. Keep its URL/title attribution and
follow its CC BY-SA/GFDL requirements when releasing derived data.

## JSONL format

Each line is one UTF-8 JSON object. Core fields are:

```json
{
  "source": "VOA Khmer",
  "id": "123456",
  "url": "https://example.org/article",
  "title": "...",
  "text": "...",
  "publication_date": "...",
  "licence": "...",
  "access_date": "2026-11-11"
}
```

Source-specific records may also contain `paragraphs` or `author`.

## Licence

The code is MIT-licensed (see `LICENSE`). Collected text keeps the licence and
rights of its original publisher and is not covered by the repository's MIT
licence. The `data/` directory is therefore excluded from Git.
