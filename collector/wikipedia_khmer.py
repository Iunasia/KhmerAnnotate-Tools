"""
Khmer Wikipedia reader (later).

Planned: read the official Khmer Wikipedia dump instead of scraping the site.

    https://dumps.wikimedia.org/kmwiki/latest/kmwiki-latest-pages-articles.xml.bz2

Steps: stream the .xml.bz2, strip wiki markup, and write one JSON object per
article to data/raw/wikipedia_km/ (same format as the VOA collector).
Wikipedia text is licensed CC BY-SA, so keep title + URL with every record.
"""


def main() -> None:
    raise NotImplementedError("Wikipedia Khmer reader is not implemented yet.")


if __name__ == "__main__":
    main()
