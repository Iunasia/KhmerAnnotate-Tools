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
| `collector/` | Collect Khmer text from VOA, Wikipedia and ODC; import authorised RFA HTML | Ready |
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
python collector/wikipedia_khmer_scraper.py --max 1000

# Process a dump that is already on disk
python collector/wikipedia_khmer_scraper.py --dump path/to/kmwiki.xml.bz2 --max 1000

# Download without extracting articles yet
python collector/wikipedia_khmer_scraper.py --download-only
```

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
