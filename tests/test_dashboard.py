import json
import tempfile
import unittest
from pathlib import Path

from baseline import data as bdata
from baseline import evaluate as beval
from checker import agreement as iaa
from checker import label_checker as lc
from checker import prepare_splits as ps
from dashboard import db as ddb
from dashboard import sync as dsync


class NeonUrlAndHelperTests(unittest.TestCase):
    def test_neon_url_normalization_and_sslmode(self):
        self.assertTrue(ddb.is_postgres_url("postgres://u:p@ep-x.neon.tech/neondb"))
        self.assertTrue(ddb.is_postgres_url("postgresql://u:p@ep-x.neon.tech/neondb"))
        self.assertFalse(ddb.is_postgres_url("data/dashboard_local.db"))
        self.assertFalse(ddb.is_postgres_url(""))

        normalized = ddb.normalize_postgres_url("postgres://u:p@ep-x.neon.tech/neondb")
        self.assertTrue(normalized.startswith("postgresql://"))
        self.assertIn("sslmode=require", normalized)

        preserved = ddb.normalize_postgres_url(
            "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=verify-full"
        )
        self.assertIn("sslmode=verify-full", preserved)
        self.assertNotIn("sslmode=require", preserved)

    def test_placeholder_adaptation_and_target_evaluation(self):
        self.assertEqual(ddb._adapt_placeholders("SELECT * FROM t WHERE a=? AND b=?", True),
                         "SELECT * FROM t WHERE a=%s AND b=%s")
        self.assertEqual(ddb._adapt_placeholders("SELECT * FROM t WHERE a=?", False),
                         "SELECT * FROM t WHERE a=?")

        self.assertEqual(ddb.evaluate_target(None, 0.70, ">="), "PENDING")
        self.assertEqual(ddb.evaluate_target(0.75, 0.70, ">="), "PASS")
        self.assertEqual(ddb.evaluate_target(0.65, 0.70, ">="), "FAIL")
        self.assertEqual(ddb.evaluate_target(0.0, 0.0, "="), "PASS")
        self.assertEqual(ddb.evaluate_target(2.0, 0.0, "="), "FAIL")


class QualityTrackerDbAndSyncTests(unittest.TestCase):
    def test_init_seed_crud_and_readiness_scorecard(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "tracker.db"
            db = ddb.QualityTrackerDB(str(db_path))
            db.init_db()
            # Re-running init_db must remain idempotent.
            db.init_db()

            targets = db.list_targets()
            self.assertEqual(len(targets), len(ddb.DEFAULT_QUALITY_TARGETS))

            # Before any runs, all 9 benchmarks are PENDING.
            initial = db.compute_readiness_scorecard()
            self.assertEqual(initial["passed"], 0)
            self.assertEqual(initial["pending"], len(ddb.DEFAULT_QUALITY_TARGETS))

            # Record passing snapshot, IAA, label audit, and XLM-R test run.
            db.record_snapshot(
                snapshot_name="v1-curated",
                raw_articles=200,
                cleaned_sentences=4867,
                segmented_sentences=4867,
                annotated_sentences=2000,
                train_sentences=1600,
                dev_sentences=200,
                test_sentences=200,
                total_entities=3100,
                per_entities=1100,
                loc_entities=1200,
                org_entities=800,
                broken_khmer_clusters=0,
                has_doc_leakage=False,
                recorded_by="Sousdey",
            )
            db.record_qa_check(
                check_type="iaa",
                batch_name="iaa_double_200",
                annotators="6 annotators",
                sentences_checked=200,
                span_f1=0.84,
                token_kappa=0.88,
                entity_only_kappa=0.82,
                passed=True,
                recorded_by="Sivming",
            )
            db.record_qa_check(
                check_type="label_audit",
                batch_name="curated_v1",
                sentences_checked=2000,
                error_count=0,
                warning_count=3,
                passed=True,
                recorded_by="Sousdey",
            )
            db.record_training_run(
                run_name="xlmr-v1-seed42",
                snapshot_name="v1-curated",
                model_name="xlm-roberta-base",
                split_evaluated="test",
                overall_precision=0.78,
                overall_recall=0.76,
                overall_f1=0.77,
                macro_f1=0.76,
                per_f1=0.81,
                loc_f1=0.77,
                org_f1=0.71,
                err_correct=240,
                err_boundary=18,
                err_type=9,
                err_missed=22,
                err_spurious=15,
                recorded_by="Sousdey",
            )
            db.add_milestone_update(
                member="Sousdey",
                stage="Stage 5: XLM-R Training & Eval",
                status="done",
                summary="First T4 GPU baseline passed 0.70 F1 target.",
            )

            final_card = db.compute_readiness_scorecard()
            self.assertEqual(final_card["passed"], len(ddb.DEFAULT_QUALITY_TARGETS))
            self.assertEqual(final_card["failed"], 0)
            self.assertEqual(final_card["readiness_pct"], 100.0)
            self.assertEqual(len(db.list_milestone_updates()), 1)

    def test_sync_cli_ingests_real_checker_and_baseline_json_files(self):
        dataset = bdata.make_synthetic_dataset(num_docs=12, sentences_per_doc=2, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db_path = root / "sync_test.db"

            # 1. Generate real IAA JSON from checker/agreement.py
            ann_a = root / "ann_a.conll"
            ann_b = root / "ann_b.conll"
            bdata.write_conll(dataset[:6], ann_a)
            bdata.write_conll(dataset[:6], ann_b)
            iaa_dir = root / "iaa"
            self.assertEqual(
                iaa.main(["--annotator-a", str(ann_a), "--annotator-b", str(ann_b), "--output-dir", str(iaa_dir)]),
                0,
            )

            # 2. Generate real QA JSON from checker/label_checker.py
            full_conll = root / "curated.conll"
            bdata.write_conll(dataset, full_conll)
            qa_dir = root / "qa"
            self.assertEqual(lc.main(["--input", str(full_conll), "--output-dir", str(qa_dir)]), 0)

            # 3. Generate real split_report.json from checker/prepare_splits.py
            split_dir = root / "curated"
            self.assertEqual(
                ps.main(["--conll", str(full_conll), "--output-dir", str(split_dir), "--seed", "42"]),
                0,
            )

            # 4. Generate real eval metrics.json from baseline/evaluate.py
            test_sents = bdata.read_conll(split_dir / "test.conll")
            pred_file = root / "test_predictions.conll"
            bdata.write_conll(test_sents, pred_file, predicted_tags=[list(s.tags) for s in test_sents])
            eval_dir = root / "eval"
            self.assertEqual(
                beval.main(["--predictions", str(pred_file), "--output-dir", str(eval_dir), "--no-charts"]),
                0,
            )

            # 5. Sync all 4 artifacts via dashboard/sync.py CLI
            self.assertEqual(
                dsync.main([
                    "--database-url", str(db_path),
                    "iaa", "--file", str(iaa_dir / "iaa_metrics.json"),
                    "--batch", "pilot_6", "--by", "Sousdey",
                ]),
                0,
            )
            self.assertEqual(
                dsync.main([
                    "--database-url", str(db_path),
                    "qa", "--file", str(qa_dir / "qa_report.json"),
                    "--batch", "curated_synth", "--by", "Sousdey",
                ]),
                0,
            )
            self.assertEqual(
                dsync.main([
                    "--database-url", str(db_path),
                    "split", "--file", str(split_dir / "split_report.json"),
                    "--name", "synth-v1", "--by", "Sousdey",
                ]),
                0,
            )
            self.assertEqual(
                dsync.main([
                    "--database-url", str(db_path),
                    "train", "--eval", str(eval_dir / "metrics.json"),
                    "--run-name", "xlmr-smoke", "--snapshot", "synth-v1", "--by", "Sousdey",
                ]),
                0,
            )
            self.assertEqual(dsync.main(["--database-url", str(db_path), "scorecard"]), 0)

            db = ddb.QualityTrackerDB(str(db_path))
            card = db.compute_readiness_scorecard()
            # 8 of 9 targets pass (only curated_sentences = 24 < 2000 fails on synthetic data).
            self.assertEqual(card["passed"], 8)
            self.assertEqual(card["failed"], 1)
            self.assertEqual(card["pending"], 0)
            self.assertEqual(db.list_training_runs()[0]["overall_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
