# KhmerAnnotate-Tools

Tools for building a **Khmer named-entity recognition (NER) dataset** with the
labels **PER** (person), **LOC** (location) and **ORG** (organisation).

The pipeline:

```
collector/  →  cleaner/  →  annotate (guidelines/)  →  checker/  →  baseline/
raw text       fix hidden     label PER/LOC/ORG          flag label     XLM-R NER
               characters                                mistakes       model
```

| Folder        | Purpose                                                     | Status   |
|---------------|-------------------------------------------------------------|----------|
| `collector/`  | Get raw Khmer text from VOA Khmer, RFA Khmer and Wikipedia  | VOA done |
| `cleaner/`    | Fix hidden-character problems (zero-width spaces, etc.)     | later    |
| `checker/`    | Flag labelling mistakes                                     | later    |
| `baseline/`   | Baseline NER model (XLM-RoBERTa)                            | later    |
| `guidelines/` | The Khmer annotation rulebook (PER, LOC, ORG + Khmer cases) | later    |
| `data/`       | Scraped output. **Not committed to git.**                   | —        |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Collect VOA Khmer text

Run these from the repo root:

```bash
# newest articles, up to 200
python -m collector.voa_khmer_scraper

# walk the whole archive, slower
python -m collector.voa_khmer_scraper --all-sitemaps --max-articles 5000 --delay 2
```

Output goes to `data/raw/voa_khmer/voa_khmer.jsonl`, one article per line:

```json
{"source": "voa_khmer", "id": "7987342", "url": "...", "title": "...",
 "published": "2025-02-25T21:54:14+07:00", "authors": ["..."],
 "categories": ["..."], "dateline": "ភ្នំពេញ —", "paragraphs": ["..."],
 "text": "...", "scraped_at": "..."}
```

How the scraper behaves:

- It finds articles through the site's sitemaps and follows `robots.txt`.
- It waits between requests (1.5 s by default; change it with `--delay`).
- It only keeps text articles. Radio and video program pages are skipped and
  their ids go into `voa_khmer.skipped.txt`.
- It can be re-run safely. Articles it has already seen are not downloaded again.
- It saves text **exactly as published**. Zero-width spaces (U+200B) are kept
  on purpose, because removing them is the job of `cleaner/`.

## License

The **code** is MIT-licensed (see [LICENSE](LICENSE)). The collected **text** is
owned by its original publishers and is not covered by this license. That is
why `data/` is kept out of git.
