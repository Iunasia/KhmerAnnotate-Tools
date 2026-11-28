import json
import tempfile
import unittest
from pathlib import Path

from baseline import data as bdata
from baseline import train_xlmr as btrain
from checker import agreement as iaa
from checker import label_checker as lc
from checker import prepare_splits as ps
from cleaner import clean_text as cleaner
from cleaner import segment_words as seg


class AgreementTests(unittest.TestCase):
    def test_cohens_kappa_and_compare_annotators(self):
        self.assertAlmostEqual(iaa.cohens_kappa(["O", "B-PER", "O"], ["O", "B-PER", "O"]), 1.0)
        s1_a = bdata.Sentence(["លោក", "សុខ", "ដារ៉ា", "នៅ", "ភ្នំពេញ", "។"], ["O", "B-PER", "I-PER", "O", "B-LOC", "O"])
        s1_b = bdata.Sentence(["លោក", "សុខ", "ដារ៉ា", "នៅ", "ភ្នំពេញ", "។"], ["O", "B-PER", "I-PER", "O", "B-ORG", "O"])
        s2 = bdata.Sentence(["ក្រសួង", "អប់រំ", "។"], ["B-ORG", "I-ORG", "O"])

        result = iaa.compare_annotators([s1_a, s2], [s1_b, s2], name_a="ann1", name_b="ann2")
        # 2 matching spans (សុខ ដារ៉ា, ក្រសួង អប់រំ) out of 3 each -> span F1 = 2/3
        self.assertAlmostEqual(result["span_f1"], 2 / 3, places=5)
        self.assertEqual(result["disagreement_sentences"], 1)
        self.assertGreater(result["token_cohens_kappa"], 0.7)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            f_a = root / "ann1.conll"
            f_b = root / "ann2.conll"
            bdata.write_conll([s1_a, s2], f_a)
            bdata.write_conll([s1_b, s2], f_b)
            out_dir = root / "iaa_out"
            code = iaa.main(
                ["--annotator-a", str(f_a), "--annotator-b", str(f_b), "--output-dir", str(out_dir)]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "iaa_metrics.json").is_file())
            self.assertTrue((out_dir / "iaa_report.html").is_file())

            # --strict should fail when span_f1 (0.6667) < 0.80.
            strict_code = iaa.main(
                [
                    "--annotator-a", str(f_a),
                    "--annotator-b", str(f_b),
                    "--output-dir", str(out_dir),
                    "--target-iaa", "0.80",
                    "--strict",
                ]
            )
            self.assertEqual(strict_code, 1)


class LabelCheckerTests(unittest.TestCase):
    def test_detects_bio_cluster_punctuation_honorific_and_inconsistency(self):
        sentences = [
            # Valid sentence: ភ្នំពេញ -> LOC
            bdata.Sentence(["នៅ", "ភ្នំពេញ", "។"], ["O", "B-LOC", "O"]),
            # Errors: broken cluster (\u17d2រ), orphan I-ORG, punctuation labeled B-PER
            # Warnings: honorific លោក in PER, inconsistent ភ្នំពេញ -> ORG
            bdata.Sentence(
                ["លោក", "សុខ", "\u17d2រ", "ភ្នំពេញ", "។"],
                ["B-PER", "I-PER", "I-ORG", "B-ORG", "B-PER"],
            ),
        ]
        report = lc.audit_sentences(sentences)
        self.assertFalse(report["passed"])
        kinds = set(report["counts_by_kind"])
        self.assertIn("broken-khmer-cluster", kinds)
        self.assertIn("invalid-bio", kinds)
        self.assertIn("punctuation-in-entity", kinds)
        self.assertIn("honorific-in-per", kinds)
        self.assertIn("inconsistent-entity", kinds)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qa.html"
            lc.write_qa_report_html(report, path)
            self.assertIn("ភ្នំពេញ", path.read_text(encoding="utf-8"))

    def test_clean_dataset_passes_cli_with_zero_exit(self):
        dataset = bdata.make_synthetic_dataset(num_docs=5, sentences_per_doc=2, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src = root / "curated.conll"
            bdata.write_conll(dataset, src)
            code = lc.main(["--input", str(src), "--output-dir", str(root / "qa")])
            self.assertEqual(code, 0)


class EndToEndStage2ToStage5Tests(unittest.TestCase):
    def test_stage2_clean_segment_to_stage4_prepare_splits_to_stage5_dry_run(self):
        # 1. Raw articles (Stage 1 format)
        raw_articles = [
            {
                "id": f"art-{i}",
                "source": "VOA Khmer",
                "url": f"https://khmer.voanews.com/a/story-{i}/{1000 + i}.html",
                "title": f"អត្ថបទ {i}",
                "licence": "Public domain",
                "text": (
                    f"លោកសុខាបានជួបពលរដ្ឋនៅភ្នំពេញសម្រាប់អត្ថបទទី {i}។ "
                    f"ក្រសួងសុខាភិបាលបានបញ្ជាក់នៅកម្ពុជាលើកទី {i}។"
                ),
            }
            for i in range(1, 16)
        ]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw_path = root / "voa_khmer.jsonl"
            raw_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_articles) + "\n",
                encoding="utf-8",
            )

            # 2. Stage 2a: clean & split sentences
            proc_dir = root / "processed"
            self.assertEqual(
                cleaner.main(["--input", str(raw_path), "--output-dir", str(proc_dir)]), 0
            )
            sent_path = proc_dir / "voa_khmer_sentences.jsonl"

            # 3. Stage 2b: segment words & export annotation batch
            batch_dir = root / "batches"
            self.assertEqual(
                seg.main(
                    [
                        "--input", str(sent_path),
                        "--output-dir", str(proc_dir),
                        "--export-dir", str(batch_dir),
                        "--smoke-size", "2",
                        "--pilot-size", "4",
                        "--pool-size", "26",
                        "--seed", "42",
                    ]
                ),
                0,
            )

            # 4. Simulate Stage 3 INCEpTION CoNLL-2002 export (NO # doc_id headers!)
            pool_txt = batch_dir / "annotation_pool_26.txt"
            pool_meta = batch_dir / "annotation_pool_26.jsonl"
            inception_conll = root / "inception_export.conll"
            with inception_conll.open("w", encoding="utf-8", newline="\n") as out_f:
                for line in pool_txt.read_text(encoding="utf-8").splitlines():
                    for tok in line.split():
                        if tok in ("ភ្នំពេញ", "កម្ពុជា"):
                            tag = "B-LOC"
                        elif tok == "ក្រសួងសុខាភិបាល":
                            tag = "B-ORG"
                        elif tok == "សុខា":
                            tag = "B-PER"
                        else:
                            tag = "O"
                        out_f.write(f"{tok}\t{tag}\n")
                    out_f.write("\n")

            # 5. Stage 4: re-attach doc_id/url provenance, run QA gate, build leak-free splits
            curated_dir = root / "curated"
            prep_code = ps.main(
                [
                    "--conll", str(inception_conll),
                    "--metadata", str(pool_meta),
                    "--output-dir", str(curated_dir),
                    "--require-metadata",
                    "--seed", "42",
                ]
            )
            self.assertEqual(prep_code, 0)
            split_report = json.loads((curated_dir / "split_report.json").read_text(encoding="utf-8"))
            self.assertEqual(split_report["provenance"]["matched_metadata"], 26)
            self.assertEqual(split_report["provenance"]["unmatched"], 0)
            self.assertFalse(split_report["leakage"]["has_doc_leakage"])

            # 6. Stage 5: hand off straight to baseline/train_xlmr.py --dry-run
            train_code = btrain.main(
                [
                    "--train", str(curated_dir / "train.conll"),
                    "--dev", str(curated_dir / "dev.conll"),
                    "--test", str(curated_dir / "test.conll"),
                    "--output-dir", str(root / "xlmr_run"),
                    "--dry-run",
                ]
            )
            self.assertEqual(train_code, 0)


if __name__ == "__main__":
    unittest.main()
