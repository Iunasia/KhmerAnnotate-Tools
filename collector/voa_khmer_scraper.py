"""
VOA Khmer scraper for KhmerAnnotate (Capstone I).

Reads VOA Khmer articles and saves text + source + date + author + licence.
No lxml needed (uses Python's built-in XML reader + html.parser).

Two ways to find articles:
  --source category   (default) walk category pages like Politics, Cambodia.
                      These list REAL news articles. Recommended.
  --source sitemap    read the sitemaps. Some batches are mostly radio
                      programs (skipped), so you get fewer real articles.

Run:
  python voa_khmer_scraper.py --max 30
  python voa_khmer_scraper.py --max 300 --source category
"""

import argparse
import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ---- EDIT THIS: a real contact email so VOA can reach your team ----
CONTACT_EMAIL = "bornsothearith@gmail.com"
USER_AGENT = f"KhmerAnnotate-CADT-Capstone/1.0 (research; {CONTACT_EMAIL})"

DELAY = 3  # seconds between requests. Be polite.
BASE = "https://khmer.voanews.com"

# Category (zone) pages that list real articles. Add/remove as you like.
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

SITEMAP_INDEX = f"{BASE}/sitemap.xml"
WANTED_SITEMAPS = ("sitemap_422_1", "sitemap_422_2", "sitemap_422_3",
                   "sitemap_422_4", "sitemap_422_news")

# Real article links look like /a/<words>/<number>.html  OR  /a/<number>.html
ARTICLE_RE = re.compile(r"/a/(?:[\w\-]+/)?\d+\.html$")

# Wire services we must NOT keep (not public domain).
WIRE_WORDS = ("Associated Press", "Reuters", "Agence France-Presse", "AFP",
              " via AP", "(AP)", "\u179a\u17c9\u1799\u1791\u17d0\u179a")

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def get(url):
    for _ in range(2):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"  ! {url} failed ({e})")
            time.sleep(DELAY)
    return None


def links_from_category(path):
    """Collect article links from one category page."""
    r = get(BASE + path)
    if not r:
        return []
    soup = BeautifulSoup(r.content, "html.parser")
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ARTICLE_RE.search(href):
            full = href if href.startswith("http") else BASE + href
            found.append(full)
    return found


def links_from_sitemaps(limit):
    r = get(SITEMAP_INDEX)
    if not r:
        return []
    root = ET.fromstring(r.content)
    subs = [loc.text.strip() for loc in root.iter("{%s}loc" % NS["sm"])
            if any(w in loc.text for w in WANTED_SITEMAPS)]
    found = []
    for sm in subs:
        r = get(sm)
        if not r:
            continue
        try:
            data = gzip.decompress(r.content)
        except OSError:
            data = r.content
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        for loc in root.iter("{%s}loc" % NS["sm"]):
            if loc.text and ARTICLE_RE.search(loc.text):
                found.append(loc.text.strip())
        time.sleep(DELAY)
        if len(found) >= limit * 4:
            break
    return found


def scrape_article(url):
    """Return a dict for one article, or None to skip it."""
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.content, "html.parser")

    # Body: <div id="article-content"> ... <div class="wsw"> paragraphs
    body = soup.find("div", id="article-content")
    if not body:
        return None
    wsw = body.find("div", class_="wsw") or body
    paragraphs = [p.get_text(strip=True) for p in wsw.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    if len(text) < 100:  # radio/video pages have almost no text
        return None

    # Wire check: skip AP/Reuters/AFP content
    if any(w in soup.get_text() for w in WIRE_WORDS):
        return None

    # Title: <h1 class="title">
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Author: <meta name="Author" content="...">
    author = ""
    m = soup.find("meta", attrs={"name": "Author"})
    if m and m.get("content"):
        author = m["content"].strip()

    # Date: <time datetime="...">
    date = ""
    t = soup.find("time")
    if t and t.get("datetime"):
        date = t["datetime"].strip()

    return {
        "url": url,
        "title": title,
        "text": text,
        "author": author,
        "publication_date": date,
        "source": "VOA Khmer",
        "licence": "Public domain (US Government / VOA) - verify not wire",
        "access_date": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--source", choices=["category", "sitemap"],
                    default="category")
    ap.add_argument("--out", default="data/raw/voa_khmer/voa_khmer.jsonl")
    args = ap.parse_args()

    # Collect candidate links
    print(f"Collecting article links (source={args.source})...")
    candidates = []
    if args.source == "category":
        for path in CATEGORY_PAGES:
            print(f"  category {path}")
            candidates.extend(links_from_category(path))
            time.sleep(DELAY)
            if len(candidates) >= args.max * 3:
                break
    else:
        candidates = links_from_sitemaps(args.max)

    # De-duplicate, keep order
    seen, unique = set(), []
    for link in candidates:
        if link not in seen:
            seen.add(link)
            unique.append(link)

    print(f"\n{len(unique)} unique links. Scraping up to {args.max}...\n")

    saved = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for link in unique:
            if saved >= args.max:
                break
            print(f"[{saved+1}] {link}")
            art = scrape_article(link)
            time.sleep(DELAY)
            if not art:
                print("     skipped (short, wire, or no body)")
                continue
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\nDone. Saved {saved} articles to {args.out}")


if __name__ == "__main__":
    main()