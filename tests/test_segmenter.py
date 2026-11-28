import json
import tempfile
import unittest
from pathlib import Path

from cleaner import clean_text as cleaner
from cleaner import segment_words as seg


class SpacedEtcFixTests(unittest.TestCase):
    def test_spaced_khmer_etc_does_not_split_sentence(self):
        text = "មានឆ្មា ឆ្កែ ។ ល។ នៅផ្ទះ។ និង ។ ល ។ ទៀត។"
        self.assertEqual(
            cleaner.split_sentences(text),
            ["មានឆ្មា ឆ្កែ ។ ល។ នៅផ្ទះ។", "និង ។ ល ។ ទៀត។"],
        )


class KhmerWordSegmenterTests(unittest.TestCase):
    def test_clusters_and_broken_token_detection(self):
        clusters = seg.split_khmer_clusters("ប្រទេសកម្ពុជាស្ត្រី")
        self.assertFalse(seg.has_broken_khmer_token(clusters))
        self.assertEqual("".join(clusters), "ប្រទេសកម្ពុជាស្ត្រី")
        # Leading coeng or dependent vowel must be flagged as broken.
        self.assertTrue(seg.has_broken_khmer_token(["\u17d2រ", "ក"]))
        self.assertTrue(seg.has_broken_khmer_token(["\u17b6ក"]))

    def test_segments_mixed_khmer_english_and_zwsp_hints(self):
        segmenter = seg.KhmerWordSegmenter()
        sentence = "លោក\u200bសុខា ធ្វើការនៅក្រសួងសុខាភិបាល ហៅកាត់ថា USAID នៅភ្នំពេញ។"
        tokens = segmenter.segment(sentence, use_zwsp_hints=True)
        self.assertFalse(seg.has_broken_khmer_token(tokens))
        self.assertTrue(all("\u200b" not in tok for tok in tokens))
        self.assertIn("លោក", tokens)
        self.assertIn("USAID", tokens)
        self.assertIn("ភ្នំពេញ", tokens)
        self.assertEqual(tokens[-1], "។")

    def test_boilerplate_and_deduplication_filter(self):
        segmenter = seg.KhmerWordSegmenter()
        records = [
            {"id": "1", "doc_id": "d1", "text": "ប្រភព៖ ក្រសួងសុខាភិបាល និង វីអូអេ"},
            {"id": "2", "doc_id": "d1", "text": "បានធ្វើបច្ចុប្បន្នភាព៖ ០៥ ធ្នូ ២០១៧"},
            {"id": "3", "doc_id": "d1", "text": "ជីវៈចម្រុះនៅកម្ពុជា"},  # short heading, no period
            {"id": "4", "doc_id": "d1", "text": "លោកសុខាទៅខេត្តកំពតដើម្បីជួបពលរដ្ឋនៅកម្ពុជា។"},
            {"id": "5", "doc_id": "d2", "text": "លោកសុខាទៅខេត្តកំពតដើម្បីជួបពលរដ្ឋនៅកម្ពុជា។"},  # duplicate
        ]
        kept, counts = seg.segment_records(records, segmenter)
        self.assertEqual(len(kept), 1)
        self.assertEqual(counts["dropped_caption-or-source"], 1)
        self.assertEqual(counts["dropped_date-line"], 1)
        self.assertEqual(counts["dropped_heading-like"], 1)
        self.assertEqual(counts["dropped_duplicate"], 1)
        self.assertEqual(counts["broken_khmer_cluster_sentences"], 0)

    def test_cli_and_batch_export_with_rotating_iaa_overlap(self):
        rows = [
            {
                "id": f"voa-{i}-0001",
                "doc_id": f"doc-{i}",
                "source": "VOA Khmer",
                "url": f"https://khmer.voanews.com/a/test/{i}.html",
                "text": f"លោកសុខាបានជួបពលរដ្ឋនៅភ្នំពេញក្នុងឆ្នាំ២០២៥ លើកទី {i}។",
            }
            for i in range(1, 31)
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inp = root / "voa_sentences.jsonl"
            inp.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            out_dir = root / "processed"
            export_dir = root / "batches"
            code = seg.main(
                [
                    "--input", str(inp),
                    "--output-dir", str(out_dir),
                    "--export-dir", str(export_dir),
                    "--smoke-size", "3",
                    "--pilot-size", "6",
                    "--pool-size", "24",
                    "--annotators", "4",
                    "--double-labeled", "8",
                    "--seed", "42",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "voa_segmented.jsonl").is_file())
            self.assertEqual(
                len((export_dir / "smoke_3.txt").read_text(encoding="utf-8").splitlines()), 3
            )
            self.assertEqual(
                len((export_dir / "pilot_6.txt").read_text(encoding="utf-8").splitlines()), 6
            )
            self.assertEqual(
                len((export_dir / "iaa_double_8.txt").read_text(encoding="utf-8").splitlines()), 8
            )
            manifest = json.loads((export_dir / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["annotator_batches"]), 4)
            # 8 overlap sentences appear twice (16) + 16 unique sentences = 32 total assignments.
            self.assertEqual(sum(manifest["annotator_batches"].values()), 32)


if __name__ == "__main__":
    unittest.main()
