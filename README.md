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
| `collector/` | Collect Khmer text from VOA and Wikipedia; import authorised RFA HTML | Ready |
| `cleaner/` | Clean hidden Unicode characters and prepare sentences | Planned |
| `guidelines/` | Khmer annotation rulebook | Planned |
| `checker/` | Flag annotation mistakes and measure agreement | Planned |
| `baseline/` | Baseline XLM-RoBERTa model | Planned |
| `data/` | Local raw and processed data (not committed) | Local only |

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

## Collect VOA Khmer news

The VOA collector checks `robots.txt`, waits between requests, skips likely
AP/Reuters/AFP material, and appends only URLs that are not already saved.

```bash
# Current articles from category pages
python collector/voa_khmer_scraper.py --max 50

# A larger collection using sitemaps
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
# Download the dump and export up to 1,000 articles
python collector/wikipedia_khmer.py --max 1000

# Process a dump that is already on disk
python collector/wikipedia_khmer.py --dump path/to/kmwiki.xml.bz2 --max 1000

# Download without extracting articles yet
python collector/wikipedia_khmer.py --download-only
```

Output: `data/raw/wikipedia_km/wikipedia_km.jsonl`

Wikipedia is encyclopaedic text, not news. Keep its URL/title attribution and
follow its CC BY-SA/GFDL requirements when releasing derived data.

## Import authorised RFA Khmer HTML

RFA's Terms of Use prohibit automated robots/scripts from accessing or copying
the website. For that reason, `rfa_khmer_scraper.py` deliberately performs no
network requests. It only parses local HTML that the team has permission to use.

```bash
python collector/rfa_khmer_scraper.py --input path/to/authorised_html --max 100
```

Output: `data/raw/rfa_khmer/rfa_khmer.jsonl`

Get written permission from RFA before automated collection or dataset release,
and keep the required attribution stored in each record.

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

Source-specific records may also contain `paragraphs`, `author`, or
`local_source_file`.

## Licence

The code is MIT-licensed (see `LICENSE`). Collected text keeps the licence and
rights of its original publisher and is not covered by the repository's MIT
licence. The `data/` directory is therefore excluded from Git.
