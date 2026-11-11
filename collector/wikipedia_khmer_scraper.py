"""Download and stream the official Khmer Wikipedia article dump into JSONL.

Wikipedia is an encyclopaedic Khmer source rather than a news source. It is
useful for entity coverage, while VOA supplies the project's news domain.

Examples (run from the repository root):

    python collector/wikipedia_khmer.py --max 1000
    python collector/wikipedia_khmer.py --dump path/to/kmwiki.xml.bz2 --max 100
    python collector/wikipedia_khmer.py --download-only
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import requests


DUMP_URL = "https://dumps.wikimedia.org/kmwiki/latest/kmwiki-latest-pages-articles.xml.bz2"
DEFAULT_DUMP = Path("data/raw/wikipedia_km/kmwiki-latest-pages-articles.xml.bz2")
DEFAULT_OUT = Path("data/raw/wikipedia_km/wikipedia_km.jsonl")
CONTACT_EMAIL = os.getenv("KHMERANNOTATE_CONTACT_EMAIL", "bornsothearith@gmail.com")
USER_AGENT = f"KhmerAnnotate-CADT-Capstone/1.0 (research; {CONTACT_EMAIL})"


def download_dump(url: str, destination: Path) -> None:
    """Download a dump atomically, keeping an incomplete .part file separate."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"  {downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                        end="\r",
                    )
    partial.replace(destination)
    print(f"\nSaved dump to {destination}")


def strip_wikicode(wikicode: str) -> str:
    """Convert MediaWiki markup to readable plain text."""
    try:
        import mwparserfromhell
    except ImportError as exc:
        raise RuntimeError(
            "mwparserfromhell is required. Run: pip install -r requirements.txt"
        ) from exc

    code = mwparserfromhell.parse(wikicode)
    text = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _open_dump(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def iter_articles(
    dump_path: Path,
    *,
    min_chars: int = 100,
    stripper: Callable[[str], str] = strip_wikicode,
) -> Iterator[dict]:
    """Stream main-namespace, non-redirect articles without loading the dump."""
    with _open_dump(dump_path) as stream:
        for _event, page in ET.iterparse(stream, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            namespace = page.tag.rsplit("}", 1)[0] + "}" if "}" in page.tag else ""
            title = page.findtext(f"{namespace}title", default="").strip()
            page_namespace = page.findtext(f"{namespace}ns", default="")
            page_id = page.findtext(f"{namespace}id", default="").strip()
            redirect = page.find(f"{namespace}redirect") is not None
            revision = page.find(f"{namespace}revision")

            if page_namespace == "0" and not redirect and revision is not None:
                timestamp = revision.findtext(f"{namespace}timestamp", default="").strip()
                raw_text = revision.findtext(f"{namespace}text", default="") or ""
                if raw_text and not raw_text.lstrip().casefold().startswith("#redirect"):
                    plain_text = stripper(raw_text)
                    if len(plain_text) >= min_chars:
                        wiki_title = quote(title.replace(" ", "_"), safe="()/:,_-")
                        yield {
                            "source": "Khmer Wikipedia",
                            "id": page_id,
                            "url": f"https://km.wikipedia.org/wiki/{wiki_title}",
                            "title": title,
                            "text": plain_text,
                            "publication_date": timestamp,
                            "licence": "CC BY-SA 4.0 and GFDL; preserve attribution and share-alike requirements",
                            "access_date": datetime.now(timezone.utc).date().isoformat(),
                        }
            page.clear()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                page_id = json.loads(line).get("id")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if page_id:
                ids.add(str(page_id))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-url", default=DUMP_URL)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max", "--max-articles", dest="max_articles", type=int)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max must be at least 1")
    if args.min_chars < 1:
        raise SystemExit("--min-chars must be at least 1")

    if not args.dump.exists():
        try:
            download_dump(args.dump_url, args.dump)
        except requests.RequestException as exc:
            raise SystemExit(f"Wikipedia dump download failed: {exc}") from exc
    if args.download_only:
        return

    known = set() if args.overwrite else existing_ids(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    saved = 0
    with args.out.open(mode, encoding="utf-8", newline="\n") as output:
        try:
            articles = iter_articles(args.dump, min_chars=args.min_chars)
            for article in articles:
                if args.max_articles is not None and saved >= args.max_articles:
                    break
                if article["id"] in known:
                    continue
                output.write(json.dumps(article, ensure_ascii=False) + "\n")
                known.add(article["id"])
                saved += 1
                if saved % 100 == 0:
                    output.flush()
                    print(f"  saved {saved} articles", end="\r")
        except (ET.ParseError, OSError, RuntimeError) as exc:
            raise SystemExit(f"Could not process Wikipedia dump: {exc}") from exc

    print(f"\nDone. Saved {saved} new Wikipedia articles to {args.out}")


if __name__ == "__main__":
    main()
