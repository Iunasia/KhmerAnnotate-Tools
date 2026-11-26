import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cleaner import clean_text as cleaner


class CleanTextTests(unittest.TestCase):
    def test_removes_hidden_characters(self):
        raw = "﻿ខ្មែរ​ជា​​ភាសា­‎។"
        counts = Counter()
        self.assertEqual(cleaner.clean_text(raw, counts=counts), "ខ្មែរជាភាសា។")
        self.assertEqual(counts["​"], 3)
        self.assertEqual(counts["﻿"], 1)

    def test_keep_zwsp(self):
        raw = "ខ្មែរ​​ជា​ ភាសា​"
        self.assertEqual(cleaner.clean_text(raw, keep_zwsp=True), "ខ្មែរ​ជា ភាសា")

    def test_spaces_and_newlines(self):
        raw = "  ក   ខ\t គ \r\n\r\n ឃ　ង  "
        self.assertEqual(cleaner.clean_text(raw), "ក ខ គ\nឃ ង")

    def test_joiners_kept_only_inside_khmer(self):
        self.assertEqual(cleaner.clean_text("ប‌ក a‍b ‌ក"), "ប‌ក ab ក")
        self.assertEqual(cleaner.clean_text("ក‌‌‌ ខ"), "ក ខ")

    def test_nfc(self):
        self.assertEqual(cleaner.clean_text("café"), "café")


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_khmer_marks_and_lines(self):
        text = "ប្រយោគទីមួយ។ ប្រយោគទីពីរ? ចប់៕\nបន្ទាត់ថ្មី"
        self.assertEqual(
            cleaner.split_sentences(text),
            ["ប្រយោគទីមួយ។", "ប្រយោគទីពីរ?", "ចប់៕", "បន្ទាត់ថ្មី"],
        )

    def test_etc_and_quotes(self):
        text = "មានឆ្មា ឆ្កែ ។ល។ នៅផ្ទះ។ គាត់ថា «ទៅ។» រួចទៅ ។"
        self.assertEqual(
            cleaner.split_sentences(text),
            ["មានឆ្មា ឆ្កែ ។ល។ នៅផ្ទះ។", "គាត់ថា «ទៅ។»", "រួចទៅ ។"],
        )

    def test_latin_full_stop_does_not_split(self):
        text = "តម្លៃ 3.5 ដុល្លារ របស់ U.S. ខ្ពស់។"
        self.assertEqual(cleaner.split_sentences(text), [text])


class SentenceRecordTests(unittest.TestCase):
    RECORD = {
        "id": "42",
        "source": "ODC",
        "url": "https://example.org/x",
        "title": "ចំណងជើង",
        "licence": "CC BY-SA 4.0",
        "text": "នេះ​ជា​ប្រយោគ​ទីមួយ។ English only. ខ្លី។\nនេះជាប្រយោគទីពីរ។",
        "html_file": "x.html",
    }

    def test_records(self):
        counts = Counter()
        records = list(cleaner.sentence_records(self.RECORD, "odc_km", counts=counts))
        self.assertEqual([r["id"] for r in records], ["odc_km-42-0001", "odc_km-42-0002"])
        self.assertEqual(records[0]["text"], "នេះជាប្រយោគទីមួយ។")
        self.assertEqual(records[1]["text"], "នេះជាប្រយោគទីពីរ។")
        self.assertEqual(records[0]["licence"], "CC BY-SA 4.0")
        self.assertIs(records[0]["changed"], True)
        self.assertNotIn("html_file", records[0])
        self.assertEqual(counts["short-sentence"], 1)

    def test_cli_writes_one_sentence_per_line(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "odc_km.jsonl"
            source.write_text(
                json.dumps(self.RECORD, ensure_ascii=False) + "\n\n", encoding="utf-8"
            )
            out_dir = Path(folder) / "out"
            code = cleaner.main(["--input", str(source), "--output-dir", str(out_dir)])
            self.assertEqual(code, 0)
            lines = (out_dir / "odc_km_sentences.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["sentence_index"], 2)


if __name__ == "__main__":
    unittest.main()
