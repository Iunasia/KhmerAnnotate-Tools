"""Import authorised, locally saved RFA Khmer article HTML into JSONL.

RFA's Terms of Use prohibit robots/scripts from accessing or copying its site,
so this program intentionally makes no network requests. Use it only with HTML
files your team has obtained with RFA's permission (or another lawful method),
and preserve the required RFA attribution in every exported record.

Example:

    python collector/rfa_khmer_scraper.py --input authorised_html --max 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup


DEFAULT_OUT = Path("data/raw/rfa_khmer/rfa_khmer.jsonl")
RFA_ATTRIBUTION = (
    "Copyright © 1998-2023, RFA. Used with the permission of Radio Free Asia, "
    "2025 M St. NW, Suite 300, Washington, D.C. 20036. https://www.rfa.org. "
    "Permitted purposes only; verify current terms and third-party rights."
)


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _news_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in _walk_json(payload):
            item_type = item.get("@type", "")
            types = item_type if isinstance(item_type, list) else [item_type]
            if any(value in {"NewsArticle", "Article", "ReportageNewsArticle"} for value in types):
                return item
    return {}


def _meta_content(soup: BeautifulSoup, *names: str) -> str:
    wanted = {name.casefold() for name in names}
    for meta in soup.find_all("meta"):
        key = (meta.get("name") or meta.get("property") or "").casefold()
        if key in wanted and meta.get("content"):
            return meta["content"].strip()
    return ""


def _author_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("name", "")).strip()
    if isinstance(value, list):
        names = [_author_name(item) for item in value]
        return ", ".join(name for name in names if name)
    return ""


def _body_paragraphs(soup: BeautifulSoup, json_ld: dict[str, Any]) -> list[str]:
    article_body = json_ld.get("articleBody")
    if isinstance(article_body, str) and len(article_body.strip()) >= 100:
        return [line.strip() for line in article_body.splitlines() if line.strip()]

    selectors = (
        "[itemprop='articleBody']",
        "[data-testid='article-body']",
        ".article-body",
        ".story-body",
        "#storytext",
        "article",
    )
    for selector in selectors:
        container = soup.select_one(selector)
        if container is None:
            continue
        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        if len("\n".join(paragraphs)) >= 100:
            return paragraphs
    return []


def extract_article(html: bytes | str, source_file: Path) -> tuple[dict | None, str]:
    """Extract a locally saved RFA HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    json_ld = _news_json_ld(soup)

    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    url = canonical.get("href", "").strip() if canonical else ""
    if not url:
        url = str(json_ld.get("url") or json_ld.get("mainEntityOfPage") or "").strip()

    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title:
        title = str(json_ld.get("headline") or "").strip()
    if not title:
        title = _meta_content(soup, "og:title", "twitter:title")

    paragraphs = _body_paragraphs(soup, json_ld)
    text = "\n".join(paragraphs)
    if not title:
        return None, "no title"
    if len(text) < 100:
        return None, "no full article body"

    author = _author_name(json_ld.get("author"))
    if not author:
        author = _meta_content(soup, "author")

    publication_date = str(json_ld.get("datePublished") or "").strip()
    if not publication_date:
        time_element = soup.find("time", datetime=True)
        publication_date = time_element["datetime"].strip() if time_element else ""
    if not publication_date:
        publication_date = _meta_content(soup, "article:published_time", "date")

    stable_value = url or f"{title}\n{text}"
    article_id = hashlib.sha256(stable_value.encode("utf-8")).hexdigest()[:16]
    return (
        {
            "source": "RFA Khmer",
            "id": article_id,
            "url": url,
            "title": title,
            "text": text,
            "paragraphs": paragraphs,
            "author": author,
            "publication_date": publication_date,
            "licence": RFA_ATTRIBUTION,
            "access_date": datetime.now(timezone.utc).date().isoformat(),
            "local_source_file": source_file.name,
        },
        "",
    )


def html_files(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in {".html", ".htm"}:
            found.append(item)
        elif item.is_dir():
            found.extend(path for path in item.rglob("*") if path.suffix.lower() in {".html", ".htm"})
        else:
            print(f"  ! input does not exist or is not HTML: {item}")
    return sorted(set(found))


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record_id = json.loads(line).get("id")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if record_id:
                ids.add(record_id)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="HTML file(s) or folder(s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max", "--max-articles", dest="max_articles", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max must be at least 1")

    files = html_files(args.input)
    if not files:
        raise SystemExit("No HTML files found. This importer does not download from RFA.")

    known = set() if args.overwrite else existing_ids(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    saved = 0
    with args.out.open(mode, encoding="utf-8", newline="\n") as stream:
        for path in files:
            if args.max_articles is not None and saved >= args.max_articles:
                break
            article, reason = extract_article(path.read_bytes(), path)
            if article is None:
                print(f"  skipped {path}: {reason}")
                continue
            if article["id"] in known:
                print(f"  already saved: {path}")
                continue
            stream.write(json.dumps(article, ensure_ascii=False) + "\n")
            known.add(article["id"])
            saved += 1

    print(f"Done. Imported {saved} authorised RFA articles to {args.out}")


if __name__ == "__main__":
    main()
