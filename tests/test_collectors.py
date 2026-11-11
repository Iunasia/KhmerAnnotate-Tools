import bz2
import json
import tempfile
import unittest
from pathlib import Path

from collector import rfa_khmer_scraper as rfa
from collector import voa_khmer_scraper as voa
from collector import wikipedia_khmer as wiki


KHMER_TEXT = "ខ្មែរជាភាសាមួយដែលមានអក្សរផ្ទាល់ខ្លួន។ " * 12


class VoaCollectorTests(unittest.TestCase):
    def test_article_url_filter(self):
        self.assertTrue(voa.is_article_url("/a/example-title/12345.html"))
        self.assertTrue(voa.is_article_url("/a/12345.html"))
        self.assertTrue(voa.is_likely_text_url("https://khmer.voanews.com/a/example-title/12345.html"))
        self.assertFalse(voa.is_likely_text_url("https://khmer.voanews.com/a/12345.html"))
        self.assertFalse(voa.is_article_url("/z/2277"))
        self.assertFalse(voa.is_article_url("https://example.com/a/12345.html"))

    def test_extract_article(self):
        html = f"""
        <html><head>
          <meta name="Author" content="VOA Khmer">
          <meta property="article:published_time" content="2026-01-02T03:04:05Z">
        </head><body>
          <h1>ចំណងជើង</h1>
          <div id="article-content"><div class="wsw">
            <p>{KHMER_TEXT}</p>
          </div></div>
        </body></html>
        """
        record, reason = voa.extract_article(html, "https://khmer.voanews.com/a/test/12345.html")
        self.assertEqual(reason, "")
        self.assertIsNotNone(record)
        self.assertEqual(record["id"], "12345")
        self.assertEqual(record["author"], "VOA Khmer")
        self.assertIn("ខ្មែរ", record["text"])

    def test_wire_article_is_skipped(self):
        html = f"""
        <h1>ចំណងជើង</h1>
        <div id="article-content"><div class="wsw">
          <p>Reuters {KHMER_TEXT}</p>
        </div></div>
        """
        record, reason = voa.extract_article(html, "https://khmer.voanews.com/a/12345.html")
        self.assertIsNone(record)
        self.assertIn("wire", reason)

    def test_parse_gzip_sitemap(self):
        xml = b"""<?xml version='1.0'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
                xmlns:image='http://www.google.com/schemas/sitemap-image/1.1'>
          <url><loc>https://khmer.voanews.com/a/123.html</loc>
            <image:image><image:loc>https://khmer.voanews.com/a/999.html</image:loc></image:image>
          </url>
        </urlset>"""
        import gzip

        root_type, locations = voa._sitemap_locations(gzip.compress(xml))
        self.assertEqual(root_type, "urlset")
        self.assertEqual(locations, ["https://khmer.voanews.com/a/123.html"])


class RfaImporterTests(unittest.TestCase):
    def test_extract_json_ld_article(self):
        payload = {
            "@type": "NewsArticle",
            "headline": "ព័ត៌មានសាកល្បង",
            "articleBody": KHMER_TEXT,
            "datePublished": "2026-02-03",
            "author": {"name": "អ្នកនិពន្ធ"},
            "url": "https://www.rfa.org/khmer/test/article.html",
        }
        html = (
            "<html><head><script type='application/ld+json'>"
            + json.dumps(payload, ensure_ascii=False)
            + "</script></head><body></body></html>"
        )
        record, reason = rfa.extract_article(html, Path("article.html"))
        self.assertEqual(reason, "")
        self.assertIsNotNone(record)
        self.assertEqual(record["title"], "ព័ត៌មានសាកល្បង")
        self.assertEqual(record["author"], "អ្នកនិពន្ធ")
        self.assertIn("permission", record["licence"].lower())


class WikipediaCollectorTests(unittest.TestCase):
    def test_streams_main_namespace_and_skips_redirect(self):
        xml = f"""<?xml version="1.0" encoding="utf-8"?>
        <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
          <page>
            <title>អត្ថបទសាកល្បង</title><ns>0</ns><id>10</id>
            <revision><id>11</id><timestamp>2026-01-01T00:00:00Z</timestamp>
              <text>{KHMER_TEXT}</text></revision>
          </page>
          <page>
            <title>Redirect</title><ns>0</ns><id>20</id><redirect title="Target" />
            <revision><id>21</id><timestamp>2026-01-01T00:00:00Z</timestamp>
              <text>#REDIRECT [[Target]]</text></revision>
          </page>
          <page>
            <title>Talk:Not an article</title><ns>1</ns><id>30</id>
            <revision><id>31</id><timestamp>2026-01-01T00:00:00Z</timestamp>
              <text>{KHMER_TEXT}</text></revision>
          </page>
        </mediawiki>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xml.bz2"
            with bz2.open(path, "wb") as stream:
                stream.write(xml.encode("utf-8"))
            records = list(
                wiki.iter_articles(path, min_chars=10, stripper=lambda value: value)
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "10")
        self.assertEqual(records[0]["title"], "អត្ថបទសាកល្បង")


if __name__ == "__main__":
    unittest.main()
