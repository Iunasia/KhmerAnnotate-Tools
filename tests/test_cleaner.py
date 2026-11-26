import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cleaner import clean_text as cleaner


class CleanTextTests(unittest.TestCase):
    def test_removes_hidden_characters(self):
        raw = "\ufeffខ្មែរ\u200bជា\u200b\u200bភាសា\u00ad\u200e។"
        counts = Counter()
        self.assertEqual(cleaner.clean_text(raw, counts=counts), "ខ្មែរជាភាសា។")
        self.assertEqual(counts["\u200b"], 3)
        self.assertEqual(counts["\ufeff"], 1)

    def test_keep_zwsp(self):
        raw = "ខ្មែរ\u200b\u200bជា\u200b ភាសា\u200b"
        self.assertEqual(cleaner.clean_text(raw, keep_zwsp=True), "ខ្មែរ\u200bជា ភាសា")

    def test_spaces_and_newlines(self):
        raw = "  ក\u00a0\u202f ខ\t គ \r\n\r\n ឃ\u3000ង  "
        self.assertEqual(cleaner.clean_text(raw), "ក ខ គ\nឃ ង")

    def test_joiners_kept_only_inside_khmer(self):
        self.assertEqual(cleaner.clean_text("ប\u200cក a\u200db \u200cក"), "ប\u200cក ab ក")
        self.assertEqual(cleaner.clean_text("ក\u200c\u200c\u200c ខ"), "ក ខ")

    def test_nfc(self):
        self.assertEqual(cleaner.clean_text("cafe\u0301"), "café")


class NormaliseKhmerTests(unittest.TestCase):
    def test_vowel_encoding(self):
        counts = Counter()
        # េ + ី -> ើ, េ + ា -> ោ, in either typing order
        self.assertEqual(cleaner.clean_text("\u1780\u17c1\u17b8 \u1780\u17b8\u17c1", counts=counts), "\u1780\u17be \u1780\u17be")
        self.assertEqual(cleaner.clean_text("\u1780\u17c1\u17b6 \u1780\u17b6\u17c1"), "\u1780\u17c4 \u1780\u17c4")
        self.assertEqual(counts["vowel-encoding"], 2)

    def test_typing_order(self):
        counts = Counter()
        # vowel typed before the subscript; subscript រ typed before another subscript
        self.assertEqual(cleaner.clean_text("\u179f\u17b8\u17d2\u179a", counts=counts), "\u179f\u17d2\u179a\u17b8")
        self.assertEqual(cleaner.clean_text("\u179f\u17d2\u179a\u17d2\u178f\u17b8"), "\u179f\u17d2\u178f\u17d2\u179a\u17b8")
        # ំ before ា, and ះ before the vowel
        self.assertEqual(cleaner.clean_text("\u1785\u17c6\u17b6 \u1793\u17c7\u17c1"), "\u1785\u17b6\u17c6 \u1793\u17c1\u17c7")
        # register shifter typed after the vowel
        self.assertEqual(cleaner.clean_text("\u179f\u17b8\u17ca"), "\u179f\u17ca\u17b8")
        self.assertEqual(counts["typing-order"], 1)

    def test_correct_text_is_unchanged(self):
        counts = Counter()
        text = "ប្រទេសកម្ពុជា ស្ត្រី ចាំ នេះ កើត ទៅ ប៉ុន្តែ ស៊ី សម្ដេច អ្នកស្រុក ក្រុមហ៊ុន"
        text = text.replace("ម្ដ", "ម្ត")
        self.assertEqual(cleaner.clean_text(text, counts=counts), text)
        self.assertEqual(sum(counts.values()), 0)

    def test_repeated_and_deprecated(self):
        self.assertEqual(cleaner.clean_text("\u1780\u17b6\u17b6 \u1780\u17d2\u17d2\u179a"), "\u1780\u17b6 \u1780\u17d2\u179a")
        self.assertEqual(cleaner.clean_text("\u17a3 \u17a4 \u1780\u17b4\u17b5"), "\u17a2 \u17a2\u17b6 \u1780")
        self.assertEqual(cleaner.clean_text("\u1780\u17c1\u17c1"), "\u1780\u17c1\u17c1")

    def test_subscript_da_ta(self):
        self.assertEqual(cleaner.clean_text("\u179f\u17d2\u178a\u17b8"), "\u179f\u17d2\u178f\u17b8")
        self.assertEqual(cleaner.clean_text("\u1781\u178e\u17d2\u178f"), "\u1781\u178e\u17d2\u178a")


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
        "text": "នេះ\u200bជា\u200bប្រយោគ\u200bទីមួយ។ English only. ខ្លី។\nនេះជាប្រយោគទីពីរ។",
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
