"""Collect ODC-original Khmer topic pages from Open Development Cambodia.

Only ``/km/topics/<slug>/`` pages are collected (optionally ``/km/profiles/``
with ``--include-profiles``). These are written by ODC and licensed CC BY-SA
4.0. News summaries (copyright of the original outlets), announcements,
updates, blog posts, maps, PDFs and the data./learn./archive. subdomains are
skipped and logged.

The collector checks robots.txt, sends one request at a time with a delay,
retries with backoff, keeps raw HTML so pages can be re-parsed offline, and
resumes where it stopped.

Examples (run from the repository root):

    python collector/odc_khmer_scraper.py --max 5
    python collector/odc_khmer_scraper.py
    python collector/odc_khmer_scraper.py --offline --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag


HOST = "opendevelopmentcambodia.net"
BASE = f"https://{HOST}"
TOPIC_INDEX = f"{BASE}/km/topics/"
SITEMAPS = {"topic": f"{BASE}/topic-sitemap.xml", "profile": f"{BASE}/profiles-sitemap.xml"}
DEFAULT_DIR = Path("data/raw/odc_km")
CONTACT_EMAIL = os.getenv("KHMERANNOTATE_CONTACT_EMAIL", "")
USER_AGENT = "KhmerAnnotate-CADT-Capstone/1.0 (Khmer NER research dataset" + (
    f"; {CONTACT_EMAIL})" if CONTACT_EMAIL else ")"
)

LICENCE = "CC BY-SA 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MIN_DELAY = 2.0
BACKOFF = (10, 30, 90)  # seconds to wait before retry 1, 2, 3
TIMEOUT = 90  # the site is slow; some pages take over 30 s

PAGE_RE = {
    "topic": re.compile(r"^/km/topics/[^/]+/$"),
    "profile": re.compile(r"^/km/profiles/[^/]+/$"),
}
NEWS_SECTIONS = ("/km/news/",)
OTHER_SECTIONS = ("/km/announcements/", "/km/updates/", "/km/blog/")
MAP_SECTIONS = ("/km/map-explorer", "/km/maps/", "/km/map/", "/km/layers/")
EXCLUDED_SUBDOMAINS = ("data.", "learn.", "archive.")
BODY_CLASS = {"topic": "single-topic", "profile": "single-profiles"}

# Reasons that will not change on a re-run, so the URL is not fetched again.
RETRYABLE_REASONS = {"download-failed"}

KHMER = re.compile(r"[ក-៿]")
LATIN = re.compile(r"[A-Za-z]")
SPACES = re.compile(r"[ \t\r\n\f\v ]+")
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
KHMER_MONTHS = {
    "មករា": 1, "កុម្ភៈ": 2, "មីនា": 3, "មេសា": 4, "ឧសភា": 5, "មិថុនា": 6,
    "កក្កដា": 7, "សីហា": 8, "កញ្ញា": 9, "តុលា": 10, "វិច្ឆិកា": 11, "ធ្នូ": 12,
}
# "ថ្ងៃទី ២៩ ខែកក្កដា ឆ្នាំ២០២៦" or the short form "០៥ ធ្នូ ២០១៧".
KHMER_DATE_RE = re.compile(
    r"(?:ថ្ងៃ\s*ទី\s*)?([0-9០-៩]{1,2})\s*(?:ខែ\s*)?([^\s0-9០-៩]+?)\s*(?:ឆ្នាំ\s*)?([0-9០-៩]{4})"
)
UPDATED_MARK = "ធ្វើបច្ចុប្បន្នភាព"  # "updated"
PUBLISHED_MARK = "ចេញផ្សាយលើកដំបូង"  # "first published"
URL_RE = re.compile(r"(?:https?://|www\.|bit\.ly/)[A-Za-z0-9\-._~:/?#@!$&'*+;=%]+")
# Broken footnote markup: when ODC's "[ref]" tag is malformed, the citation
# (sometimes a news headline) is shown inline, ending in "[/ref]".
BROKEN_REF_RE = re.compile(r"\[/?ref\]|ref\]")
BRACKET_RE = re.compile(r"\[[^\[\]]{0,120}\]")
THIRD_PARTY_RE = re.compile(r"third[\s-]*party|ភាគី\s*ទី\s*បី", re.IGNORECASE)
REMOVE_SELECTORS = (
    ".reference-footnote", "sup", ".wp-caption", "figure", "figcaption", "img",
    "picture", "iframe", "script", "style", "noscript", "table", "blockquote",
    "form", ".post-meta", ".main-title", ".swp-hidden-panel-wrap", ".swp_social_panel", ".swp-content-locator",
    ".flourish-embed", ".odm-summary", ".share-widget",
)
# "Related to ...", "References", "Related topics": navigation, not prose.
DROP_HEADING_PREFIXES = ("ទាក់ទងនឹង", "ឯកសារយោង", "ប្រធានបទពាក់ព័ន្ធ", "On this page")


# --------------------------------------------------------------------------
# URLs


def normalise_url(url: str, base: str = BASE) -> str:
    """Absolute https URL without query/fragment; ODC page paths end in '/'."""
    parsed = urlparse(urljoin(base, url.strip()))
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path or "/"
    if host == HOST and path.startswith("/km/") and "." not in path.rsplit("/", 1)[-1]:
        path = path if path.endswith("/") else path + "/"
    return f"https://{host}{path}"


def classify_url(url: str) -> tuple[str | None, str]:
    """Return (page type, "") for collectable pages or (None, skip reason).

    The reason is "" for links that are simply out of scope (menus, external
    sites, category listings) and are not worth logging.
    """
    parsed = urlparse(url)
    host, path = parsed.netloc.lower(), parsed.path
    if host != HOST:
        if host.endswith("." + HOST) and host.startswith(EXCLUDED_SUBDOMAINS):
            return None, "excluded-subdomain"
        return None, ""
    if path.lower().endswith(".pdf"):
        return None, "pdf"
    if path.startswith(NEWS_SECTIONS):
        return None, "news-summary"
    if path.startswith(OTHER_SECTIONS):
        return None, "excluded-section"
    if path.startswith(MAP_SECTIONS):
        return None, "map-page"
    for page_type, pattern in PAGE_RE.items():
        if pattern.match(path):
            return page_type, ""
    return None, ""


def url_slug(url: str) -> str:
    return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]


def html_path(html_dir: Path, url: str) -> Path:
    section = urlparse(url).path.split("/")[2]  # topics | profiles
    return html_dir / section / f"{url_slug(url)}.html"


# --------------------------------------------------------------------------
# Parsing helpers


def _decode(html: bytes | str) -> str | bytes:
    """Prefer UTF-8 so Khmer is never mis-guessed as a legacy code page."""
    if isinstance(html, str):
        return html
    try:
        return html.decode("utf-8-sig")
    except UnicodeDecodeError:
        return html


def parse_khmer_date(text: str) -> str:
    """'ថ្ងៃទី ២៩ ខែកក្កដា ឆ្នាំ២០២៦' -> '2026-07-29' ('' if not found)."""
    match = KHMER_DATE_RE.search(text.replace("​", ""))
    if not match:
        return ""
    day, month_name, year = match.groups()
    month = KHMER_MONTHS.get(month_name.strip())
    if month is None:
        return ""
    try:
        return datetime(
            int(year.translate(KHMER_DIGITS)), month, int(day.translate(KHMER_DIGITS))
        ).date().isoformat()
    except ValueError:
        return ""


def take_date_notes(container: Tag) -> tuple[str, str, str]:
    """Remove short "first published ... / updated: ..." notes from the body.

    Returns (date_published ISO, last_updated ISO, raw update text). Must run
    before <sup> tags are stripped: some notes are wrapped in <sup>.
    """
    published = updated = updated_raw = ""
    for paragraph in container.find_all("p"):
        flat = _clean(paragraph.get_text("").replace("​", ""))
        if len(flat) > 300 or (UPDATED_MARK not in flat and PUBLISHED_MARK not in flat):
            continue
        before, mark, after = flat.partition(UPDATED_MARK)
        note_published = parse_khmer_date(before) if PUBLISHED_MARK in before else ""
        note_updated = parse_khmer_date(after) if mark else ""
        if not (note_published or note_updated):
            continue  # ordinary prose that happens to mention updating
        published = published or note_published
        if note_updated and not updated:
            updated, updated_raw = note_updated, mark + after
        paragraph.decompose()
    return published, updated, updated_raw


def khmer_stats(text: str) -> tuple[int, float]:
    """(Khmer character count, Khmer share of Khmer+Latin letters)."""
    khmer = len(KHMER.findall(text))
    latin = len(LATIN.findall(text))
    return khmer, (khmer / (khmer + latin) if khmer + latin else 0.0)


def _clean(text: str) -> str:
    # Zero-width spaces (U+200B) are kept; the cleaner/ step handles them.
    return SPACES.sub(" ", text).strip()


def _is_link_only(element: Tag) -> bool:
    letters = lambda value: len(re.sub(r"[\W_]", "", value.replace("​", "")))
    link_text = "".join(a.get_text("") for a in element.find_all("a"))
    return bool(link_text.strip()) and letters(element.get_text("")) <= letters(link_text)


def _is_caption(text: str) -> bool:
    return len(text) < 300 and (
        "CC BY" in text or text.startswith(("រូបភាព", "Photo", "Image"))
    )


def _drop_heading_sections(container: Tag) -> None:
    """Remove 'related topics' and 'references' headings and what follows them."""
    headings = ("h1", "h2", "h3", "h4", "h5", "h6")
    for heading in container.find_all(headings):
        if heading.decomposed:
            continue
        title = heading.get_text("").replace("​", "").strip()
        if not title.startswith(DROP_HEADING_PREFIXES):
            continue
        for sibling in list(heading.find_next_siblings()):
            if sibling.name in headings:
                break
            sibling.decompose()
        heading.decompose()


def prose_paragraphs(container: Tag) -> list[str]:
    """Main Khmer prose (paragraphs and plain-text bullets) from a body element."""
    for selector in REMOVE_SELECTORS:
        for element in container.select(selector):
            element.decompose()
    _drop_heading_sections(container)
    for br in container.find_all("br"):
        br.replace_with(" ")

    paragraphs: list[str] = []
    for element in container.find_all(["p", "li"]):
        if element.name == "li" and element.find("p"):
            continue  # its <p> children are taken individually
        if _is_link_only(element):
            continue
        # Khmer has no spaces between words, so join inline nodes directly.
        text = _clean(element.get_text(""))
        if not text or _is_caption(text):
            continue
        label = text.replace("​", "")
        if len(label) < 80 and label.startswith(DROP_HEADING_PREFIXES):
            continue  # a section label written as a paragraph
        if BROKEN_REF_RE.search(text):
            continue  # inline citation from broken [ref] markup
        for note in BRACKET_RE.findall(text):
            if "ឯកសារយោង" in note.replace("​", ""):  # "[same reference as above]"
                text = text.replace(note, "")
        if URL_RE.search(text):
            text = _clean(re.sub(r"\(\s*\)", "", URL_RE.sub("", text)))
            if len(text.replace("​", "")) < 20:
                continue  # e.g. "Website: http://..." with only the label left
        khmer, share = khmer_stats(text)
        if khmer and share >= 0.3:
            paragraphs.append(text)
    return list(dict.fromkeys(paragraphs))


def body_links(html: bytes | str, base_url: str) -> list[str]:
    """Normalised links found inside the article body of a topic page."""
    soup = BeautifulSoup(_decode(html), "lxml")
    body = soup.select_one("div.post-content")
    if body is None:
        return []
    for ref in body.select(".reference-footnote"):
        ref.decompose()
    links = []
    for anchor in body.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        links.append(normalise_url(href, base_url))
    return list(dict.fromkeys(links))


# --------------------------------------------------------------------------
# Page extraction


def extract_page(
    html: bytes | str,
    url: str,
    *,
    include_profiles: bool = False,
    min_khmer: int = 200,
    min_profile_khmer: int = 500,
    scraped_at: str = "",
) -> tuple[dict | None, str, str]:
    """Return (record, skip reason, detail) for one downloaded ODC page."""
    page_type, reason = classify_url(url)
    if page_type is None:
        return None, reason or "out-of-scope", url
    if page_type == "profile" and not include_profiles:
        return None, "profile-disabled", "use --include-profiles"

    soup = BeautifulSoup(_decode(html), "lxml")
    body_classes = set(soup.body.get("class", [])) if soup.body else set()
    if "single-news-article" in body_classes or soup.select_one("a[href*='/news_source/']"):
        return None, "news-summary", "news post type / news_source link"
    if BODY_CLASS[page_type] not in body_classes:
        post_type = next((c for c in body_classes if c.startswith("single-")), "unknown")
        return None, "not-odc-original", f"expected {BODY_CLASS[page_type]}, got {post_type}"

    article = soup.select_one("article#content") or soup.select_one("section#content")
    if article is None:
        return None, "no-article-body", "no #content element"
    if THIRD_PARTY_RE.search(article.get_text(" ").replace("​", "")):
        return None, "third-party", "page marked as third-party content"

    heading = soup.find("h1")
    title = _clean(heading.get_text("")) if heading else ""
    categories = [
        _clean(a.get_text("")) for a in soup.select(".post-meta li.categories a")
    ]
    # The pencil date in the page header is WordPress's "modified" date.
    date_item = soup.select_one(".post-meta li.date")
    modified_raw = _clean(date_item.get_text("")) if date_item else ""

    if page_type == "topic":
        body = article.select_one("div.post-content")
        if body is None:
            return None, "no-article-body", "no div.post-content"
        minimum = min_khmer
    else:
        body = article
        for h1 in body.find_all("h1"):
            h1.decompose()
        minimum = min_profile_khmer
    # A page's own "updated: <date>" note wins over the header date.
    date_published, note_updated, note_raw = take_date_notes(body)
    last_updated_raw = note_raw if note_updated else modified_raw

    # Language check on the whole body (before English paragraphs are
    # filtered out) so an untranslated page is logged as non-Khmer.
    for element in body.select(".reference-footnote, .main-title, .post-meta, script, style"):
        element.decompose()
    body_text = body.get_text(" ")
    body_khmer, body_share = khmer_stats(body_text)
    if body_khmer + len(LATIN.findall(body_text)) >= 200 and body_share < 0.5:
        return None, "non-khmer", f"Khmer share {body_share:.0%}"

    text = "\n".join(prose_paragraphs(body))
    khmer, _ = khmer_stats(text)
    if khmer < minimum:
        reason = "profile-without-prose" if page_type == "profile" else "too-short"
        return None, reason, f"{khmer} Khmer characters (< {minimum})"

    post_id = next(
        (c.removeprefix("postid-") for c in body_classes if c.startswith("postid-")), ""
    )
    return (
        {
            "source": "ODC",
            "id": post_id,
            "page_type": page_type,
            "url": url,
            "title": title,
            "date_published": date_published,
            "last_updated": parse_khmer_date(last_updated_raw),
            "last_updated_raw": last_updated_raw,
            "page_modified": parse_khmer_date(modified_raw),
            "categories": categories,
            "text": text,
            "licence": LICENCE,
            "licence_url": LICENCE_URL,
            "attribution": (
                f'"{title}" ({url}) by Open Development Cambodia (ODC), '
                f"licensed under {LICENCE}"
            ),
            "scraped_at": scraped_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "changed": False,
        },
        "",
        "",
    )


# --------------------------------------------------------------------------
# Network


class Fetcher:
    """One request at a time, at least ``delay`` seconds apart, with backoff."""

    def __init__(self, delay: float, timeout: float = TIMEOUT) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "km,en;q=0.5"})
        self.delay = delay
        self.timeout = timeout
        self._last = 0.0

    def _wait(self) -> None:
        remaining = self.delay - (time.monotonic() - self._last)
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str) -> tuple[requests.Response | None, str]:
        reason = ""
        for attempt in range(len(BACKOFF) + 1):
            self._wait()
            retry_after = None
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                reason = type(exc).__name__
            else:
                status = response.status_code
                if status == 200:
                    return response, ""
                # WordPress on ODC often renders the full page, then fails in
                # the footer and sends 500. Accept it when the article is there.
                if status == 500 and b'id="content"' in response.content:
                    return response, "http 500 (page rendered)"
                reason = f"http {status}"
                if status in (401, 403, 404, 410):
                    return None, reason
                retry_after = response.headers.get("Retry-After", "")
            finally:
                self._last = time.monotonic()
            if attempt < len(BACKOFF):
                wait = BACKOFF[attempt]
                if retry_after and retry_after.isdigit():
                    wait = min(max(wait, int(retry_after)), 300)
                print(f"  ! {reason}; retry {attempt + 1}/{len(BACKOFF)} in {wait}s")
                time.sleep(wait)
        return None, reason


def load_robots(fetcher: Fetcher) -> RobotFileParser:
    """Read robots.txt the way urllib.robotparser does, but fail closed on errors.

    ODC currently returns 404, which the robots.txt standard treats as
    "everything allowed". Network failures and 5xx stop the run.
    """
    robots_url = f"{BASE}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    fetcher._wait()
    try:
        response = fetcher.session.get(robots_url, timeout=fetcher.timeout)
    finally:
        fetcher._last = time.monotonic()
    if response.status_code == 200:
        parser.parse(response.text.splitlines())
    elif response.status_code in (401, 403):
        parser.disallow_all = True
    elif 400 <= response.status_code < 500:
        parser.allow_all = True
    else:
        raise requests.HTTPError(f"robots.txt returned HTTP {response.status_code}")
    return parser


def index_links(fetcher: Fetcher, robots: RobotFileParser, page_types: set[str]) -> list[str]:
    """Candidate pages from the Khmer topic index and the sitemaps."""
    found: list[str] = []
    if robots.can_fetch(USER_AGENT, TOPIC_INDEX):
        response, reason = fetcher.get(TOPIC_INDEX)
        if response is None:
            print(f"  ! could not read topic index: {reason}")
        else:
            soup = BeautifulSoup(_decode(response.content), "lxml")
            for anchor in soup.find_all("a", href=True):
                url = normalise_url(anchor["href"], TOPIC_INDEX)
                if classify_url(url)[0] in page_types:
                    found.append(url)
    for page_type in sorted(page_types):
        sitemap = SITEMAPS[page_type]
        if not robots.can_fetch(USER_AGENT, sitemap):
            print(f"  ! robots.txt disallows {sitemap}")
            continue
        response, reason = fetcher.get(sitemap)
        if response is None:
            print(f"  ! could not read {sitemap}: {reason}")
            continue
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            print(f"  ! invalid sitemap {sitemap}: {exc}")
            continue
        section = "topics" if page_type == "topic" else "profiles"
        for loc in root.iterfind("{*}url/{*}loc"):
            # The sitemap lists English URLs; the Khmer page shares the slug.
            if loc.text:
                found.append(f"{BASE}/km/{section}/{url_slug(loc.text)}/")
    return list(dict.fromkeys(found))


# --------------------------------------------------------------------------
# Output files


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


class SkipLog:
    def __init__(self, path: Path, overwrite: bool) -> None:
        previous = [] if overwrite else read_jsonl(path)
        self.logged = {row["url"] for row in previous}
        self.permanent = {
            row["url"] for row in previous if row.get("reason") not in RETRYABLE_REASONS
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w" if overwrite else "a", encoding="utf-8", newline="\n")
        self.counts: dict[str, int] = {}

    def add(self, url: str, reason: str, detail: str = "", *, quiet: bool = False) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if not quiet:
            print(f"  skipped ({reason}): {detail or url}")
        row = {
            "url": url,
            "reason": reason,
            "detail": detail,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.logged.add(url)
        if reason not in RETRYABLE_REASONS:
            self.permanent.add(url)


# --------------------------------------------------------------------------
# Main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max", "--max-pages", dest="max_pages", type=int, default=None,
                        help="stop after saving this many new pages (default: all)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR,
                        help="folder for odc_km.jsonl, skipped.jsonl and html/")
    parser.add_argument("--delay", type=float, default=3.0,
                        help=f"seconds between requests (minimum {MIN_DELAY})")
    parser.add_argument("--include-profiles", action="store_true",
                        help="also collect /km/profiles/ pages that contain real prose")
    parser.add_argument("--offline", action="store_true",
                        help="re-parse saved HTML only; make no network requests")
    parser.add_argument("--overwrite", action="store_true",
                        help="rewrite odc_km.jsonl and skipped.jsonl (saved HTML is kept)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        raise SystemExit("--max must be at least 1")
    if args.delay < MIN_DELAY:
        raise SystemExit(f"--delay must be at least {MIN_DELAY} seconds")

    out_path = args.out_dir / "odc_km.jsonl"
    html_dir = args.out_dir / "html"
    page_types = {"topic", "profile"} if args.include_profiles else {"topic"}
    skips = SkipLog(args.out_dir / "skipped.jsonl", args.overwrite)
    saved_urls = set() if args.overwrite else {row.get("url") for row in read_jsonl(out_path)}

    fetcher = robots = None
    if args.offline:
        candidates = sorted(
            f"{BASE}/km/{path.parent.name}/{path.stem}/" for path in html_dir.glob("*/*.html")
        )
        candidates = [url for url in candidates if classify_url(url)[0] in page_types]
        print(f"Offline: {len(candidates)} saved HTML pages.")
    else:
        fetcher = Fetcher(args.delay)
        try:
            robots = load_robots(fetcher)
        except requests.RequestException as exc:
            raise SystemExit(f"Could not read ODC robots.txt; stopping safely: {exc}") from exc
        print("Collecting ODC links from the Khmer topic index and sitemaps...")
        candidates = index_links(fetcher, robots, page_types)
        print(f"Found {len(candidates)} candidate pages.")

    queue = deque(candidates)
    queued = set(candidates)
    saved = examined = 0
    with out_path.open("w" if args.overwrite else "a", encoding="utf-8", newline="\n") as out:
        while queue and (args.max_pages is None or saved < args.max_pages):
            url = queue.popleft()
            if url in saved_urls or url in skips.permanent:
                continue
            examined += 1
            print(f"[{examined}, saved {saved}] {url}")

            cached = html_path(html_dir, url)
            if cached.exists():
                html = cached.read_bytes()
            elif args.offline:
                continue
            elif not robots.can_fetch(USER_AGENT, url):
                skips.add(url, "robots", "disallowed by robots.txt")
                continue
            else:
                response, note = fetcher.get(url)
                if response is None:
                    skips.add(url, "download-failed", note)
                    continue
                if note:
                    print(f"  note: {note}")
                html = response.content
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(html)
            scraped_at = datetime.fromtimestamp(cached.stat().st_mtime, timezone.utc)

            if classify_url(url)[0] == "topic":
                for link in body_links(html, url):
                    link_type, reason = classify_url(link)
                    if link_type in page_types and link not in queued:
                        queued.add(link)
                        queue.append(link)
                    elif reason and link not in skips.logged:
                        skips.add(link, reason, f"linked from {url}", quiet=True)

            record, reason, detail = extract_page(
                html,
                url,
                include_profiles=args.include_profiles,
                scraped_at=scraped_at.isoformat(timespec="seconds"),
            )
            if record is None:
                skips.add(url, reason, detail)
                continue
            record["html_file"] = cached.relative_to(args.out_dir).as_posix()
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            saved_urls.add(url)
            saved += 1
            print(f"  saved: {record['title']} ({len(record['text'])} chars)")

    skips.stream.close()
    print(f"\nDone. Saved {saved} new pages to {out_path}")
    if skips.counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(skips.counts.items()))
        print(f"Skipped this run: {summary} (details in {args.out_dir / 'skipped.jsonl'})")


if __name__ == "__main__":
    main()
