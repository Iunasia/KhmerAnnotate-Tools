import bz2
import json
import tempfile
import unittest
from pathlib import Path

from collector import odc_khmer_scraper as odc
from collector import rfa_khmer_scraper as rfa
from collector import voa_khmer_scraper as voa
from collector import wikipedia_khmer_scraper as wiki


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

    def test_utf8_bytes_without_charset_meta(self):
        # The CLI passes raw file bytes. Without a <meta charset>, BeautifulSoup
        # guesses the encoding; this body was mis-detected as ibm855.
        body = "ភាសា" + KHMER_TEXT
        payload = {
            "@type": "NewsArticle",
            "headline": "ព័ត៌មានសាកល្បង",
            "articleBody": body,
            "author": {"name": "អ្នកនិពន្ធ"},
        }
        html = (
            "<html><head><script type='application/ld+json'>"
            + json.dumps(payload, ensure_ascii=False)
            + "</script></head><body></body></html>"
        ).encode("utf-8")
        for data in (html, b"\xef\xbb\xbf" + html):
            # Deterministic check, independent of the charset detector version.
            self.assertEqual(rfa._decode_html(data), html.decode("utf-8"))
            record, reason = rfa.extract_article(data, Path("article.html"))
            self.assertEqual(reason, "")
            self.assertEqual(record["title"], "ព័ត៌មានសាកល្បង")
            self.assertEqual(record["author"], "អ្នកនិពន្ធ")
            self.assertTrue(record["text"].startswith("ភាសាខ្មែរ"))


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


ODC_TOPIC = "https://opendevelopmentcambodia.net/km/topics/biodiversity/"


def odc_page(body_class="single single-topic postid-97791", content="", meta=""):
    return f"""<html><body class="{body_class}">
      <header><nav><a href="/km/news/">ព័ត៌មាន</a></nav></header>
      <article id="content" class="single-post">
        <section class="container section-title main-title"><h1>ជីវៈចម្រុះ</h1>
          <div class="post-meta"><ul>
            <li class="date"><i class="fa fa-pencil"></i> ថ្ងៃទី ៩ ខែវិច្ឆិកា ឆ្នាំ២០២០ </li>
            <li class="categories"><a href="/km/category/x/">បរិស្ថាន</a></li>
            {meta}
          </ul></div></section>
        <section class="container"><section class="content">
          <div class="post-content">{content}</div>
        </section></section>
      </article>
      <footer><p>{KHMER_TEXT} ប្រភព​ភាគី​ទី​បី</p></footer>
    </body></html>"""


ODC_CONTENT = f"""
  <p>{KHMER_TEXT}<sup><a class="reference_footnote" href="#ref-1">1</a></sup></p>
  <div class="wp-caption"><img src="a.jpg"><p class="wp-caption-text">រូបភាព៖ ថតដោយ X អាជ្ញាប័ណ្ណ CC BY-NC-ND 2.0</p></div>
  <blockquote><p>“សម្រង់ពីប្រភពផ្សេង” {KHMER_TEXT}</p></blockquote>
  <table><tr><td>តារាង {KHMER_TEXT}</td></tr></table>
  <ul><li>ការ​នេសាទ​ហួស​កម្រិត</li><li><a href="/km/topics/forest-classifications/">ចំណាត់ថ្នាក់ព្រៃឈើ</a></li></ul>
  <p><em>បាន​ធ្វើ​បច្ចុប្បន្នភាព​៖ ០៥ ធ្នូ ២០១៧</em></p>
  <p><strong>ប្រធានបទ​ពាក់ព័ន្ធ</strong></p>
  <p>{KHMER_TEXT} “ចំណងជើង”, កាសែតភ្នំពេញបុស្ដិ៍, ២០១៦។ <a href="http://www.phnompenhpost.com/x">http://www.phnompenhpost.com/x</a>[/ref]</p>
  <p>គេហទំព័រ: <a href="http://www.nbc.org.kh">http://www.nbc.org.kh</a></p>
  <p>ក្រុមហ៊ុន​ (http://example.org/a) ថា៖ ខ្មែរ។[ដូចឯកសារយោងខាងលើ។]</p>
  <h4>ទាក់ទងនឹងជីវៈចម្រុះ</h4>
  <p>ទាក់ទង {KHMER_TEXT}</p>
  <div class="reference-footnote"><h4>ឯកសារយោង</h4><ol><li>ឯកសារ {KHMER_TEXT}
    <a href="https://www.phnompenhpost.com/x">PPP</a></li></ol></div>
  <a href="/km/news/some-story/">x</a> <a href="/wp-content/uploads/r.pdf">pdf</a>
  <a href="https://data.opendevelopmentcambodia.net/km/dataset/x">data</a>
"""


class OdcCollectorTests(unittest.TestCase):
    def test_classify_url(self):
        base = "https://opendevelopmentcambodia.net"
        self.assertEqual(odc.classify_url(f"{base}/km/topics/biodiversity/"), ("topic", ""))
        self.assertEqual(odc.classify_url(f"{base}/km/profiles/x/"), ("profile", ""))
        self.assertEqual(odc.classify_url(f"{base}/km/news/x/"), (None, "news-summary"))
        self.assertEqual(odc.classify_url(f"{base}/km/blog/x/"), (None, "excluded-section"))
        self.assertEqual(odc.classify_url(f"{base}/km/map-explorer/"), (None, "map-page"))
        self.assertEqual(odc.classify_url(f"{base}/files/r.PDF"), (None, "pdf"))
        self.assertEqual(
            odc.classify_url("https://archive.opendevelopmentcambodia.net/x"),
            (None, "excluded-subdomain"),
        )
        self.assertEqual(odc.classify_url(f"{base}/topics/biodiversity/"), (None, ""))
        self.assertEqual(
            odc.normalise_url("/km/topics/x?queried_post_type=topic#top"),
            f"{base}/km/topics/x/",
        )

    def test_parse_khmer_date(self):
        self.assertEqual(odc.parse_khmer_date("ថ្ងៃទី ២៩ ខែកក្កដា ឆ្នាំ២០២៦"), "2026-07-29")
        self.assertEqual(odc.parse_khmer_date("ថ្ងៃទី០៤ ខែ វិច្ឆិកាឆ្នាំ២០១៥"), "2015-11-04")
        self.assertEqual(odc.parse_khmer_date("បាន​ធ្វើ​បច្ចុប្បន្នភាព៖ ០៥ ធ្នូ ២០១៧"), "2017-12-05")
        self.assertEqual(odc.parse_khmer_date("no date"), "")

    def test_topic_keeps_only_prose(self):
        record, reason, _ = odc.extract_page(odc_page(content=ODC_CONTENT), ODC_TOPIC)
        self.assertEqual(reason, "")
        self.assertEqual(record["title"], "ជីវៈចម្រុះ")
        self.assertEqual(record["categories"], ["បរិស្ថាន"])
        self.assertEqual(record["source"], "ODC")
        self.assertEqual(record["licence"], "CC BY-SA 4.0")
        self.assertIs(record["changed"], False)
        # The page's own "updated" note wins over the header date.
        self.assertEqual(record["last_updated"], "2017-12-05")
        self.assertEqual(record["page_modified"], "2020-11-09")
        paragraphs = record["text"].split("\n")
        self.assertEqual(
            paragraphs,
            [KHMER_TEXT.strip(), "ការ​នេសាទ​ហួស​កម្រិត", "ក្រុមហ៊ុន​ ថា៖ ខ្មែរ។"],
        )
        for unwanted in ("CC BY", "សម្រង់", "តារាង", "ឯកសារ", "ទាក់ទង", "ធ្វើ", "PPP", "http", "ref]", "បុស្ដិ៍"):
            self.assertNotIn(unwanted, record["text"])

    def test_news_summary_is_skipped(self):
        page = odc_page(body_class="single single-news-article", content=ODC_CONTENT)
        self.assertEqual(odc.extract_page(page, ODC_TOPIC)[1], "news-summary")
        page = odc_page(content=ODC_CONTENT, meta='<li><a href="/km/news_source/kiripost/">គិរីផូស</a></li>')
        self.assertEqual(odc.extract_page(page, ODC_TOPIC)[1], "news-summary")

    def test_other_skips(self):
        self.assertEqual(
            odc.extract_page(odc_page(body_class="single single-post"), ODC_TOPIC)[1],
            "not-odc-original",
        )
        english = odc_page(content="<p>" + "English only text about forests. " * 30 + "ខ្មែរ</p>")
        self.assertEqual(odc.extract_page(english, ODC_TOPIC)[1], "non-khmer")
        self.assertEqual(odc.extract_page(odc_page(content="<p>ខ្លី</p>"), ODC_TOPIC)[1], "too-short")
        marked = odc_page(content=f"<p>Third-party content. {KHMER_TEXT}</p>")
        self.assertEqual(odc.extract_page(marked, ODC_TOPIC)[1], "third-party")
        profile = "https://opendevelopmentcambodia.net/km/profiles/x/"
        self.assertEqual(odc.extract_page(odc_page(), profile)[1], "profile-disabled")

    def test_profile_dates_and_prose(self):
        content = (
            "<p><em><sup>ទំព័រ​នេះ​ត្រូវ​បាន​ចេញ​ផ្សាយ​លើក​ដំបូង​នៅថ្ងៃ​ទី១២ ខែមករា ឆ្នាំ​២០២២ "
            "និងបានធ្វើបច្ចុប្បន្នភាពនៅថ្ងៃទី ២២ ខែមេសា ឆ្នាំ២០២៥</sup></em></p>"
            f"<p>{KHMER_TEXT}</p>"
        )
        page = odc_page(body_class="single single-profiles postid-1", content=content)
        url = "https://opendevelopmentcambodia.net/km/profiles/x/"
        record, reason, _ = odc.extract_page(page, url, include_profiles=True, min_profile_khmer=100)
        self.assertEqual(reason, "")
        self.assertEqual(record["date_published"], "2022-01-12")
        self.assertEqual(record["last_updated"], "2025-04-22")
        self.assertEqual(record["text"], KHMER_TEXT.strip())

    def test_body_links_skip_references(self):
        links = odc.body_links(odc_page(content=ODC_CONTENT), ODC_TOPIC)
        self.assertIn("https://opendevelopmentcambodia.net/km/topics/forest-classifications/", links)
        self.assertIn("https://opendevelopmentcambodia.net/km/news/some-story/", links)
        self.assertNotIn("https://www.phnompenhpost.com/x", links)


if __name__ == "__main__":
    unittest.main()
