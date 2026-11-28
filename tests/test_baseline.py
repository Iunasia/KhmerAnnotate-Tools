import json
import tempfile
import unittest
from pathlib import Path

from baseline import data as bdata
from baseline import evaluate as beval
from baseline import metrics as bmetrics
from baseline import predict as bpredict
from baseline import train_xlmr as btrain


class FakeBatchEncoding(dict):
    """Minimal stand-in for a Hugging Face fast tokenizer batch output."""

    def __init__(self, input_ids, attention_mask, word_id_lists):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_id_lists = word_id_lists

    def word_ids(self, batch_index: int = 0):
        return self._word_id_lists[batch_index]


class FakeSubwordTokenizer:
    """Simulates SentencePiece splitting each Khmer word into 2 subwords."""

    def __call__(self, token_lists, *, is_split_into_words=True, truncation=True, max_length=256):
        assert is_split_into_words
        all_ids = []
        all_masks = []
        all_word_ids = []
        for words in token_lists:
            ids = [0]  # <s>
            wids = [None]
            for word_index, word in enumerate(words):
                pieces = 2 if len(word) > 2 else 1
                for piece_idx in range(pieces):
                    if truncation and len(ids) >= max_length - 1:
                        break
                    ids.append(10 + word_index * 10 + piece_idx)
                    wids.append(word_index)
                if truncation and len(ids) >= max_length - 1:
                    break
            ids.append(2)  # </s>
            wids.append(None)
            all_ids.append(ids)
            all_masks.append([1] * len(ids))
            all_word_ids.append(wids)
        return FakeBatchEncoding(all_ids, all_masks, all_word_ids)


class ConllAndJsonlTests(unittest.TestCase):
    def test_conll_roundtrip_with_metadata_and_docstart(self):
        raw = (
            "-DOCSTART- -X- O O\n\n"
            "# doc_id = voa-001\n"
            "# sent_id = voa-001-01\n"
            "# url = https://khmer.voanews.com/a/example/1.html\n"
            "លោក\tO\n"
            "សុខ\tB-PER\n"
            "ដារ៉ា\tI-PER\n"
            "នៅ\tO\n"
            "ភ្នំពេញ\tB-LOC\n"
            "។\tO\n\n"
            "ក្រសួង B-ORG\n"
            "សុខាភិបាល I-ORG\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "sample.conll"
            source.write_text(raw, encoding="utf-8")
            sentences = bdata.read_conll(source)
            self.assertEqual(len(sentences), 2)
            self.assertEqual(sentences[0].doc_id, "voa-001")
            self.assertEqual(sentences[0].sent_id, "voa-001-01")
            self.assertEqual(sentences[0].tags, ["O", "B-PER", "I-PER", "O", "B-LOC", "O"])
            # Second sentence inherits doc_id from previous header until changed.
            self.assertEqual(sentences[1].doc_id, "voa-001")
            self.assertEqual(sentences[1].tokens, ["ក្រសួង", "សុខាភិបាល"])

            target = Path(folder) / "out.conll"
            bdata.write_conll(sentences, target)
            reloaded = bdata.read_conll(target)
            self.assertEqual(
                [(s.tokens, s.tags, s.doc_id) for s in reloaded],
                [(s.tokens, s.tags, s.doc_id) for s in sentences],
            )

    def test_read_jsonl_dataset(self):
        line = {
            "id": "s1",
            "doc_id": "d1",
            "tokens": ["ធនាគារ", "ជាតិ", "កម្ពុជា"],
            "tags": ["B-ORG", "I-ORG", "I-ORG"],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")
            sentences = bdata.load_dataset_file(path)
            self.assertEqual(len(sentences), 1)
            self.assertEqual(sentences[0].doc_id, "d1")
            self.assertEqual(sentences[0].tags, ["B-ORG", "I-ORG", "I-ORG"])


class BioValidationAndSplittingTests(unittest.TestCase):
    def test_validate_and_repair_bio(self):
        good = bdata.Sentence(["សុខ", "ដារ៉ា", "។"], ["B-PER", "I-PER", "O"])
        bad = bdata.Sentence(
            ["សុខ\u200b", "ដារ៉ា", "ភ្នំពេញ", "។"],
            ["I-PER", "I-LOC", "B-UNKNOWN", "O"],
        )
        issues = bdata.validate_sentences([good, bad])
        kinds = [issue.kind for issue in issues]
        self.assertIn("invisible-unicode", kinds)
        self.assertIn("invalid-bio", kinds)
        self.assertIn("unknown-label", kinds)

        self.assertEqual(
            bdata.repair_bio_tags(["I-PER", "I-PER", "O", "I-LOC", "I-ORG"]),
            ["B-PER", "I-PER", "O", "B-LOC", "B-ORG"],
        )

    def test_document_split_has_zero_leakage(self):
        dataset = bdata.make_synthetic_dataset(num_docs=20, sentences_per_doc=3, seed=42)
        train, dev, test = bdata.split_by_document(dataset, seed=42)
        self.assertEqual(len(train) + len(dev) + len(test), len(dataset))
        self.assertGreater(len(train), len(dev))
        self.assertGreater(len(dev), 0)
        self.assertGreater(len(test), 0)

        leakage = bdata.verify_no_leakage(train, dev, test)
        self.assertFalse(leakage["has_doc_leakage"])

        # Deterministic for the same seed.
        train2, dev2, test2 = bdata.split_by_document(dataset, seed=42)
        self.assertEqual([s.sent_id for s in train], [s.sent_id for s in train2])
        self.assertEqual([s.sent_id for s in dev], [s.sent_id for s in dev2])
        self.assertEqual([s.sent_id for s in test], [s.sent_id for s in test2])

        # Explicitly verify leakage detector catches shared doc_id.
        leaked = bdata.verify_no_leakage(train, [train[0]], test)
        self.assertTrue(leaked["has_doc_leakage"])


class SubwordAlignmentTests(unittest.TestCase):
    def test_align_labels_first_subword_only_and_all_tokens(self):
        word_tags = ["O", "B-PER", "I-PER", "B-LOC"]
        word_ids = [None, 0, 1, 1, 2, 2, 3, None]

        first_only = bdata.align_labels_with_word_ids(word_tags, word_ids, label_all_tokens=False)
        self.assertEqual(
            first_only,
            [
                -100,
                bdata.LABEL2ID["O"],
                bdata.LABEL2ID["B-PER"],
                -100,
                bdata.LABEL2ID["I-PER"],
                -100,
                bdata.LABEL2ID["B-LOC"],
                -100,
            ],
        )

        all_tokens = bdata.align_labels_with_word_ids(word_tags, word_ids, label_all_tokens=True)
        # Continuation subword of B-PER becomes I-PER (not duplicate B-PER).
        self.assertEqual(all_tokens[2], bdata.LABEL2ID["B-PER"])
        self.assertEqual(all_tokens[3], bdata.LABEL2ID["I-PER"])

    def test_decode_word_predictions_and_truncation(self):
        word_ids = [None, 0, 1, 1, 2, None]
        # Second subword of word 1 predicts a noisy tag; first subword must win.
        pred_ids = [
            bdata.LABEL2ID["O"],
            bdata.LABEL2ID["O"],
            bdata.LABEL2ID["B-ORG"],
            bdata.LABEL2ID["B-LOC"],
            bdata.LABEL2ID["I-ORG"],
            bdata.LABEL2ID["O"],
        ]
        tags, truncated = bdata.decode_word_predictions(pred_ids, word_ids, num_words=4)
        self.assertEqual(tags, ["O", "B-ORG", "I-ORG", "O"])
        self.assertEqual(truncated, 1)

    def test_encode_sentences_reports_truncation(self):
        sent = bdata.Sentence(
            tokens=["លោក", "សុខ", "ដារ៉ា", "ធ្វើការ", "នៅ", "ភ្នំពេញ"],
            tags=["O", "B-PER", "I-PER", "O", "O", "B-LOC"],
        )
        encoded, stats = bdata.encode_sentences([sent], FakeSubwordTokenizer(), max_length=7)
        self.assertEqual(stats["truncated_sentences"], 1)
        self.assertGreater(stats["truncated_words"], 0)
        self.assertEqual(len(encoded[0]["input_ids"]), len(encoded[0]["labels"]))


class MetricsAndErrorAnalysisTests(unittest.TestCase):
    def test_extract_spans_standard_and_strict(self):
        tags = ["O", "B-PER", "I-PER", "I-LOC", "B-ORG"]
        self.assertEqual(
            bmetrics.extract_spans(tags, strict_iob2=False),
            [
                bmetrics.EntitySpan("PER", 1, 3),
                bmetrics.EntitySpan("LOC", 3, 4),
                bmetrics.EntitySpan("ORG", 4, 5),
            ],
        )
        self.assertEqual(
            bmetrics.extract_spans(tags, strict_iob2=True),
            [
                bmetrics.EntitySpan("PER", 1, 3),
                bmetrics.EntitySpan("ORG", 4, 5),
            ],
        )

    def test_compute_metrics_and_error_categories(self):
        s1 = bdata.Sentence(
            tokens=["លោក", "សុខ", "ដារ៉ា", "នៅ", "ធនាគារ", "ជាតិ", "ក្នុង", "ភ្នំពេញ", "។"],
            tags=["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
            doc_id="doc-1",
            sent_id="doc-1-01",
        )
        # Predictions:
        # - "លោក សុខ ដារ៉ា" -> boundary error on PER (includes title លោក)
        # - "ធនាគារ ជាតិ" -> type error (predicted LOC instead of ORG)
        # - "ភ្នំពេញ" -> missed (O)
        # - "។" -> spurious ORG
        pred_s1 = ["B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O", "B-ORG"]

        s2 = bdata.Sentence(
            tokens=["សៀមរាប", "ស្អាត", "។"],
            tags=["B-LOC", "O", "O"],
            doc_id="doc-2",
            sent_id="doc-2-01",
        )
        pred_s2 = ["B-LOC", "O", "O"]

        metrics = bmetrics.compute_ner_metrics([s1.tags, s2.tags], [pred_s1, pred_s2])
        # Overall: 1 TP (សៀមរាប), 3 FP, 3 FN -> P=0.25, R=0.25, F1=0.25
        self.assertAlmostEqual(metrics["overall"]["precision"], 0.25)
        self.assertAlmostEqual(metrics["overall"]["recall"], 0.25)
        self.assertAlmostEqual(metrics["overall"]["f1"], 0.25)
        self.assertEqual(metrics["per_entity"]["LOC"]["tp"], 1)

        errors = bmetrics.analyze_errors([s1, s2], [pred_s1, pred_s2])
        self.assertEqual(errors["counts"]["correct"], 1)
        self.assertEqual(errors["counts"]["boundary_error"], 1)
        self.assertEqual(errors["counts"]["type_error"], 1)
        self.assertEqual(errors["counts"]["missed"], 1)
        self.assertEqual(errors["counts"]["spurious"], 1)
        self.assertEqual(errors["confusion_matrix"]["ORG"]["LOC"], 1)
        self.assertEqual(errors["confusion_matrix"]["LOC"]["O"], 1)
        self.assertEqual(errors["confusion_matrix"]["O"]["ORG"], 1)

        with tempfile.TemporaryDirectory() as folder:
            report = bmetrics.write_error_report_html(
                metrics, errors, Path(folder) / "report.html"
            )
            html_text = report.read_text(encoding="utf-8")
            self.assertIn("សុខ ដារ៉ា", html_text)
            self.assertIn("ent-PER", html_text)

    def test_decode_word_probs(self):
        # 2 words: word 0 split into 2 subwords, word 1 has 1 subword
        probs = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # <s>
            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],  # word 0 piece 0 -> B-PER
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # word 0 piece 1 (ignored)
            [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],  # word 1 -> O
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # </s>
        ]
        tags, confs, entropies = bpredict.decode_word_probs(
            probs, [None, 0, 0, 1, None], num_words=2
        )
        self.assertEqual(tags, ["B-PER", "O"])
        self.assertAlmostEqual(confs[0], 0.8)
        self.assertGreater(entropies[0], entropies[1])


class BaselineCliTests(unittest.TestCase):
    def test_train_dry_run_with_auto_split_and_evaluate_predictions(self):
        dataset = bdata.make_synthetic_dataset(num_docs=15, sentences_per_doc=2, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            merged = root / "all.conll"
            out_dir = root / "run"
            bdata.write_conll(dataset, merged)

            code = btrain.main(
                ["--input", str(merged), "--output-dir", str(out_dir), "--dry-run"]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "splits" / "train.conll").is_file())
            self.assertTrue((out_dir / "splits" / "dev.conll").is_file())
            self.assertTrue((out_dir / "splits" / "test.conll").is_file())

            # Write a 3-column predictions file and evaluate without PyTorch.
            test_sents = bdata.read_conll(out_dir / "splits" / "test.conll")
            preds = [list(s.tags) for s in test_sents]
            preds[0] = ["O"] * len(preds[0])
            pred_file = root / "test_predictions.conll"
            bdata.write_conll(test_sents, pred_file, predicted_tags=preds)

            eval_dir = root / "eval"
            eval_code = beval.main(
                ["--predictions", str(pred_file), "--output-dir", str(eval_dir), "--no-charts"]
            )
            self.assertEqual(eval_code, 0)
            metrics_json = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_json["sentences"], len(test_sents))
            self.assertLess(metrics_json["metrics"]["overall"]["f1"], 1.0)
            self.assertTrue((eval_dir / "error_report.html").is_file())

    def test_train_dry_run_rejects_document_leakage(self):
        dataset = bdata.make_synthetic_dataset(num_docs=4, sentences_per_doc=2, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_path = root / "train.conll"
            dev_path = root / "dev.conll"
            bdata.write_conll(dataset, train_path)
            # Reuse doc from train in dev -> must fail closed.
            bdata.write_conll([dataset[0]], dev_path)
            with self.assertRaises(SystemExit):
                btrain.main(
                    ["--train", str(train_path), "--dev", str(dev_path), "--dry-run"]
                )


if __name__ == "__main__":
    unittest.main()
