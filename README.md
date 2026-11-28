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
| `collector/` | Collect Khmer text from VOA, Wikipedia and ODC | Ready (488 raw articles collected) |
| `cleaner/` | Clean hidden characters, fix Khmer typing order/vowels, split sentences, segment words (`segment_words.py`) | Ready |
| `guidelines/` | Khmer NER annotation rulebook and local Docker INCEpTION setup | Ready |
| `checker/` | Measure IAA (`agreement.py`), flag label errors (`label_checker.py`), build leak-free splits (`prepare_splits.py`) | Ready |
| `baseline/` | Baseline XLM-RoBERTa NER model (CoNLL-2002 train, eval, charts, HTML report) | Ready (waiting for annotated data) |
| `data/` | Local raw and processed data (not committed) | Shared on Hugging Face |

## How we annotate

Each article is split into sentences, and each sentence is annotated word by
word with `PER`, `LOC`, `ORG` or no label. Khmer is written without spaces
between words, so the text must also be split into words before annotation.
That step is not built yet.

`CLAUDE.md` holds the current state of the project (what is done and what is
still open) and is updated after each piece of work.

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

## Collect Open Development Cambodia (ODC) Khmer topic pages

`collector/odc_khmer_scraper.py` collects ODC's own Khmer topic pages
(`/km/topics/<slug>/`). It finds them from the Khmer topic index and ODC's
topic sitemap, then follows sub-topic links found inside each page.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1

# Test run: save 5 pages
python collector/odc_khmer_scraper.py --max 5

# Full run (about 10-15 minutes; stop with Ctrl+C and re-run to resume)
python collector/odc_khmer_scraper.py

# Also collect /km/profiles/ pages that contain real prose
python collector/odc_khmer_scraper.py --include-profiles

# Re-parse the saved HTML after changing the extraction code (no network)
python collector/odc_khmer_scraper.py --offline --overwrite
```

Output in `data/raw/odc_km/`:

| File | Contents |
|---|---|
| `odc_km.jsonl` | One page per line: `url`, `title`, `date_published`, `last_updated`, `page_modified`, `categories`, `text`, `source` (`"ODC"`), `licence` (`"CC BY-SA 4.0"`), `attribution`, `scraped_at`, `changed` (`false`; set to `true` after sentence splitting) |
| `skipped.jsonl` | Every skipped page or link with its reason (`robots`, `news-summary`, `pdf`, `non-khmer`, `excluded-section`, `excluded-subdomain`, `map-page`, `not-odc-original`, `third-party`, `too-short`, `download-failed`, ...) |
| `html/` | Raw HTML of every downloaded page, so pages can be re-parsed offline |

`text` keeps only the main Khmer prose: paragraphs and plain-text bullet
points. The numbered reference list, footnote numbers, images, photo captions,
tables, quotations, "related topics" link lists, menus and the footer are
removed. Zero-width spaces are kept for the `cleaner/` step. `last_updated`
comes from the page's own "បានធ្វើបច្ចុប្បន្នភាព" ("updated") note when it
has one, otherwise from the date in the page header. `page_modified` is always
the header date.

Already-saved URLs and permanently skipped URLs are not fetched again, so the
scraper can be stopped and resumed. Pages that failed to download are retried
on the next run. `--overwrite` rewrites the JSONL files but keeps the saved HTML.

### ODC licence rules (follow these)

Based on ODC's [Terms of Use](https://opendevelopmentcambodia.net/terms-of-use/):

1. Collect **only original content written by ODC** (CC BY-SA 4.0): topic
   pages, plus profiles that contain real prose.
2. **Skip news-summary posts.** They are copyrighted by their original sources
   (e.g. Khmer Times, Phnom Penh Post). The scraper skips `/km/news/` and any
   page whose WordPress post type is a news article or that names a news
   source. It also skips anything marked as third-party content.
3. **Skip PDFs and reports.** Some use CC BY-NC-SA, which must not be mixed in.
   Announcements, updates, blog posts, map pages and the `data.`, `learn.` and
   `archive.` subdomains are also skipped.
4. **Respect robots.txt.** The scraper checks it before running and before
   every page. ODC currently returns 404, which the robots.txt standard treats
   as "everything allowed". If robots.txt cannot be reached, the scraper stops.
5. **Scrape slowly.** One request at a time, at least 2 seconds apart (3 by
   default), with a User-Agent naming the project and retries with backoff
   (10 s, 30 s, 90 s). The ODC server is slow and sometimes returns HTTP 500
   after rendering a full page; those pages are kept only if the article is
   present.

Every released record must keep this credit line:

> Contains content from Open Development Cambodia (opendevelopmentcambodia.net),
> licensed under CC BY-SA 4.0. Text was split into sentences and annotated.

ODC's terms also ask for the specific page title and link. Each record's
`attribution` field holds that, for example
`"ជីវៈចម្រុះ" (https://opendevelopmentcambodia.net/km/topics/biodiversity/) by
Open Development Cambodia (ODC), licensed under CC BY-SA 4.0`. Because of
ShareAlike, data derived from ODC text must also be released under CC BY-SA 4.0.

## Clean the text and split it into sentences

`cleaner/clean_text.py` reads the raw article files and writes one sentence
per line, ready for annotation.

```bash
# Clean every data/raw/*/*.jsonl file
python cleaner/clean_text.py

# Clean one file
python cleaner/clean_text.py --input data/raw/voa_khmer/voa_khmer.jsonl

# Keep zero-width spaces as word-boundary hints
python cleaner/clean_text.py --keep-zwsp
```

Output: `data/processed/<name>_sentences.jsonl`, for example
`data/processed/odc_km_sentences.jsonl`.

What it does:

1. Applies Unicode NFC normalisation.
2. Removes hidden characters: zero-width spaces, byte-order marks, soft
   hyphens, direction marks and control characters. Zero-width joiners and
   non-joiners are kept only between two Khmer characters, where they change
   how the text is drawn.
3. Turns unusual spaces (no-break, thin, tabs, ...) into a normal space and
   collapses repeated spaces.
4. Fixes Khmer vowel encoding: a vowel typed as two parts (`េ` + `ី` for
   `ើ`, `េ` + `ា` for `ោ`) becomes the single correct character, and
   deprecated characters are replaced.
5. Fixes Khmer typing order: the marks of each syllable are put in the
   standard order (subscripts, register shifter, vowel, final signs), and a
   mark typed twice in a row is kept once. Subscript `ដ` and subscript `ត`
   look identical, so they are stored as `្ត`, except after `ណ` (`ណ្ដ`).
   After this, words that look the same are stored the same way.
6. Splits each paragraph into sentences at `។`, `៕`, `៚`, `?` and `!`. The
   abbreviation `។ល។` and Latin full stops do not end a sentence.
7. Drops sentences with fewer than 10 Khmer characters (`--min-khmer`), such
   as English-only lines.

The script prints how many of each fix it made.

Each output line has `id` (`<file>-<article id>-<sentence number>`), `doc_id`,
`sentence_index`, `text`, the article's `source`, `url`, `title`, `licence`
and `attribution` (when present), and `changed: true`.

Not done yet:

- Splitting sentences into words. The zero-width spaces in ODC text mark word
  boundaries, so use `--keep-zwsp` if you need them.
- The output still contains some lines that are not sentences (headings, date
  lines, "source:" captions) and some repeated sentences.
- The subscript `ដ`/`ត` rule has not been checked against a reference yet.
  Review it before relying on the output.
- The cleaner has only been run on the ODC data so far.

Run the tests with `python -m unittest discover -s tests -t .`.

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
