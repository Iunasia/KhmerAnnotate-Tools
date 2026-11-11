"""Collect original Khmer text articles from VOA Khmer.

The collector discovers articles from category pages or VOA's sitemap, checks
robots.txt, skips likely wire-service material, and writes one JSON object per
line. Existing URLs are not downloaded again unless ``--overwrite`` is used.

Examples (run from the repository root):

    python collector/voa_khmer_scraper.py --max 50
    python collector/voa_khmer_scraper.py --source sitemap --max 500
    python collector/voa_khmer_scraper.py --overwrite --max 20
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


BASE = "https://khmer.voanews.com"
SITEMAP_INDEX = f"{BASE}/sitemap.xml"
DEFAULT_OUT = Path("data/raw/voa_khmer/voa_khmer.jsonl")
CONTACT_EMAIL = os.getenv("KHMERANNOTATE_CONTACT_EMAIL", "bornsothearith@gmail.com")
USER_AGENT = f"KhmerAnnotate-CADT-Capstone/1.0 (research; {CONTACT_EMAIL})"

CATEGORY_PAGES = [
    "/z/2277",  # Cambodia
    "/z/2278",  # Politics
    "/z/2279",  # Economy
    "/z/2967",  # Human rights
    "/z/2282",  # Health
    "/z/2280",  # Education
    "/z/2281",  # Environment
    "/z/2886",  # Southeast Asia
]

ARTICLE_RE = re.compile(r"^/a/(?:[\w-]+/)?\d+\.html$")
WIRE_WORDS = (
    "associated press",
    "reuters",
    "agence france-presse",
    "afp",
    " via ap",
    "(ap)",
    "រ៉យទ័រ",
)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def get(
    session: requests.Session,
    url: str,
    *,
    delay: float,
    attempts: int = 2,
) -> requests.Response | None:
    """Download a URL with a bounded retry count."""
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            print(f"  ! attempt {attempt}/{attempts} failed for {url}: {exc}")
            if attempt < attempts:
                time.sleep(delay)
    return None


def load_robots(session: requests.Session) -> RobotFileParser:
    """Fetch VOA's robots.txt and fail closed if it cannot be read."""
    robots_url = f"{BASE}/robots.txt"
    response = session.get(robots_url, timeout=30)
    response.raise_for_status()
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    return parser


def is_article_url(url: str) -> bool:
    parsed = urlparse(urljoin(BASE, url))
    return parsed.netloc == urlparse(BASE).netloc and bool(ARTICLE_RE.match(parsed.path))


def is_likely_text_url(url: str) -> bool:
    """Slugged VOA URLs are much more often full text than numeric-only URLs."""
    return bool(re.match(r"^/a/[\w-]+/\d+\.html$", urlparse(url).path))


def links_from_category(
    session: requests.Session,
    robots: RobotFileParser,
    path: str,
    *,
    delay: float,
) -> list[str]:
    """Collect article links from one VOA category page."""
    url = urljoin(BASE, path)
    if not robots.can_fetch(USER_AGENT, url):
        print(f"  ! robots.txt disallows {url}")
        return []
    response = get(session, url, delay=delay)
    if response is None:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    found: list[str] = []
    for anchor in soup.find_all("a", href=True):
        full = urljoin(BASE, anchor["href"])
        if is_article_url(full):
            found.append(full)
    return found


def _sitemap_locations(content: bytes) -> tuple[str, list[str]]:
    """Return (root type, locations) for plain or gzip-compressed sitemap XML."""
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    root = ET.fromstring(content)
    root_type = root.tag.rsplit("}", 1)[-1]
    locations: list[str] = []
    for entry in root:
        # Use the primary <loc> directly under <url>/<sitemap>. Nested media
        # namespaces can contain additional URLs that are not article pages.
        for element in entry:
            if element.tag.rsplit("}", 1)[-1] == "loc" and element.text:
                locations.append(element.text.strip())
                break
    return root_type, locations


def links_from_sitemaps(
    session: requests.Session,
    robots: RobotFileParser,
    limit: int,
    *,
    delay: float,
    max_sitemaps: int = 100,
) -> list[str]:
    """Walk VOA sitemap indexes until enough article candidates are found."""
    queue = [SITEMAP_INDEX]
    visited: set[str] = set()
    found: list[str] = []
    target = max(limit * 10, 100)

    while queue and len(visited) < max_sitemaps and len(found) < target:
        sitemap_url = queue.pop(0)
        if sitemap_url in visited:
            continue
        visited.add(sitemap_url)
        if not robots.can_fetch(USER_AGENT, sitemap_url):
            print(f"  ! robots.txt disallows {sitemap_url}")
            continue
        response = get(session, sitemap_url, delay=delay)
        if response is None:
            continue
        try:
            root_type, locations = _sitemap_locations(response.content)
        except (ET.ParseError, OSError) as exc:
            print(f"  ! invalid sitemap {sitemap_url}: {exc}")
            continue

        if root_type == "sitemapindex":
            def sitemap_priority(item: str) -> tuple[int, str]:
                name = item.lower()
                # VOA's 422 batches contain Khmer service pages. The generic
                # news sitemap is largely radio/program pages without text.
                if "sitemap_422_" in name and "news" not in name:
                    return (0, item)
                if "sitemap_422_news" in name:
                    return (1, item)
                return (2, item)

            locations.sort(key=sitemap_priority)
            queue.extend(location for location in locations if location not in visited)
        else:
            article_locations = [
                location for location in locations if is_article_url(location)
            ]
            article_locations.sort(key=lambda item: not is_likely_text_url(item))
            found.extend(article_locations[: target - len(found)])
        time.sleep(delay)

    return found


def _meta_content(soup: BeautifulSoup, *names: str) -> str:
    wanted = {name.casefold() for name in names}
    for meta in soup.find_all("meta"):
        key = (meta.get("name") or meta.get("property") or "").casefold()
        if key in wanted and meta.get("content"):
            return meta["content"].strip()
    return ""


def extract_article(html: bytes | str, url: str) -> tuple[dict | None, str]:
    """Extract one VOA article, returning (record, skip reason)."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", id="article-content")
    if body is None:
        return None, "no article body"

    content = body.find("div", class_="wsw") or body
    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    text = "\n".join(paragraphs)
    if len(text) < 100:
        return None, "article text is too short"

    author = _meta_content(soup, "author")
    wire_haystack = f"{author}\n{text[:1000]}".casefold()
    if any(word in wire_haystack for word in WIRE_WORDS):
        return None, "possible AP/Reuters/AFP wire content"

    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title:
        title = _meta_content(soup, "og:title", "twitter:title")

    publication_date = ""
    time_element = soup.find("time", datetime=True)
    if time_element:
        publication_date = time_element["datetime"].strip()
    if not publication_date:
        publication_date = _meta_content(
            soup, "article:published_time", "date", "datepublished"
        )

    match = re.search(r"/(\d+)\.html$", urlparse(url).path)
    article_id = match.group(1) if match else ""
    return (
        {
            "source": "VOA Khmer",
            "id": article_id,
            "url": url,
            "title": title,
            "text": text,
            "paragraphs": paragraphs,
            "author": author,
            "publication_date": publication_date,
            "licence": "Public domain only when produced exclusively by VOA; wire content excluded",
            "access_date": datetime.now(timezone.utc).date().isoformat(),
        },
        "",
    )


def scrape_article(
    session: requests.Session,
    robots: RobotFileParser,
    url: str,
    *,
    delay: float,
) -> tuple[dict | None, str]:
    if not robots.can_fetch(USER_AGENT, url):
        return None, "disallowed by robots.txt"
    response = get(session, url, delay=delay)
    if response is None:
        return None, "download failed"
    return extract_article(response.content, url)


def existing_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    urls: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                url = json.loads(line).get("url")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if url:
                urls.add(url)
    return urls


def unique_in_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", "--max-articles", dest="max_articles", type=int, default=50)
    parser.add_argument("--source", choices=("category", "sitemap"), default="category")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_articles < 1:
        raise SystemExit("--max must be at least 1")
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")

    session = make_session()
    try:
        robots = load_robots(session)
    except requests.RequestException as exc:
        raise SystemExit(f"Could not read VOA robots.txt; stopping safely: {exc}") from exc

    print(f"Collecting VOA links (source={args.source})...")
    if args.source == "category":
        candidates: list[str] = []
        for path in CATEGORY_PAGES:
            print(f"  category {path}")
            candidates.extend(
                links_from_category(session, robots, path, delay=args.delay)
            )
            time.sleep(args.delay)
    else:
        candidates = links_from_sitemaps(
            session, robots, args.max_articles, delay=args.delay
        )

    candidates = unique_in_order(candidates)
    already_saved = set() if args.overwrite else existing_urls(args.out)
    candidates = [url for url in candidates if url not in already_saved]
    print(f"Found {len(candidates)} new unique candidate links.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    saved = 0
    examined = 0
    with args.out.open(mode, encoding="utf-8", newline="\n") as stream:
        for url in candidates:
            if saved >= args.max_articles:
                break
            examined += 1
            print(f"[candidate {examined}, saved {saved}/{args.max_articles}] {url}")
            article, reason = scrape_article(session, robots, url, delay=args.delay)
            time.sleep(args.delay)
            if article is None:
                print(f"  skipped: {reason}")
                continue
            stream.write(json.dumps(article, ensure_ascii=False) + "\n")
            stream.flush()
            saved += 1

    print(f"Done. Saved {saved} new articles to {args.out}")


if __name__ == "__main__":
    main()
