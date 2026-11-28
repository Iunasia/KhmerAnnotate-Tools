"""KhmerAnnotate Quality & Training Tracker — Streamlit + Neon PostgreSQL Dashboard.

Tracks expected Capstone targets vs actual pipeline results across Stages 1–5
so the whole team (and advisors) can inspect dataset provenance, zero-leakage
audits, IAA scores, label QA errors, and XLM-R NER training runs in one shared
place backed by Neon PostgreSQL.

Run locally:
    streamlit run dashboard/app.py

Deploy on Streamlit Community Cloud + Neon:
    Set ``DATABASE_URL = "postgresql://...neon.tech/neondb?sslmode=require"``
    in Streamlit Secrets (see ``dashboard/README.md``).
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from dashboard.db import QualityTrackerDB
from dashboard.sync import (
    ingest_iaa_payload,
    ingest_qa_payload,
    ingest_split_payload,
    ingest_training_payload,
)

TEAM_MEMBERS = [
    "Sousdey (Model Lead)",
    "Chansothearith (Project Lead)",
    "Sivming (Rulebook Lead)",
    "Meas (Platform Lead)",
    "Darren (Data Source Lead)",
    "Vibolrathana (Tools Lead)",
]

STAGES = [
    "Stage 1: Collector",
    "Stage 2: Cleaner & Segmenter",
    "Stage 3: Guidelines & INCEpTION",
    "Stage 4: Quality Gate (IAA & Splits)",
    "Stage 5: XLM-R Training & Eval",
]

st.set_page_config(
    page_title="KhmerAnnotate — Quality & Training Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_database_url() -> str:
    """Read DATABASE_URL from Streamlit secrets first, then environment variables."""
    try:
        secret_url = st.secrets.get("DATABASE_URL", "")
        if secret_url:
            return str(secret_url)
    except Exception:
        pass
    return os.getenv("DATABASE_URL", "")


@st.cache_resource
def get_db(database_url: str) -> QualityTrackerDB:
    db = QualityTrackerDB(database_url or None)
    db.init_db()
    return db


def _status_badge(status: str) -> str:
    return {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "PENDING": "PENDING",
    }.get(status, status)


def _to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main() -> None:
    db_url = _resolve_database_url()
    db = get_db(db_url)
    scorecard = db.compute_readiness_scorecard()

    # Sidebar: deployment status & quick refresh
    with st.sidebar:
        st.title("KhmerAnnotate")
        st.caption("Capstone Group 01 — NER Quality & Training Tracker")
        if db.is_postgres:
            st.success("Connected: Neon PostgreSQL")
        else:
            st.info(f"Local Mode: SQLite\n`{db.sqlite_path}`")
            st.caption("Set `DATABASE_URL` in Streamlit Secrets to sync with Neon PostgreSQL.")

        st.divider()
        st.metric("Capstone Readiness", f"{scorecard['readiness_pct']:.1f}%")
        st.progress(min(max(scorecard["readiness_pct"] / 100.0, 0.0), 1.0))
        st.caption(
            f"Passed: {scorecard['passed']} | Failed: {scorecard['failed']} | Pending: {scorecard['pending']}"
        )
        if st.button("Refresh Data"):
            st.rerun()

    st.title("KhmerAnnotate — Final Quality & Training Dashboard")
    st.caption(
        "Shared live tracker comparing desired Capstone I benchmarks against recorded "
        "dataset integrity, IAA agreement, label QA audits, and XLM-R NER evaluations."
    )

    # Top KPI Strip
    latest_run = scorecard["latest_run"]
    latest_iaa = scorecard["latest_iaa"]
    latest_qa = scorecard["latest_qa"]
    latest_snap = scorecard["latest_snapshot"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Target Readiness",
            f"{scorecard['passed']}/{scorecard['total_targets']}",
            f"{scorecard['readiness_pct']:.0f}% ready",
        )
    with c2:
        if latest_run:
            delta_f1 = latest_run["overall_f1"] - 0.70
            st.metric(
                "Latest Overall F1",
                f"{latest_run['overall_f1']:.4f}",
                f"{delta_f1:+.4f} vs 0.70",
            )
        else:
            st.metric("Latest Overall F1", "Pending", "Target >= 0.70")
    with c3:
        if latest_iaa and latest_iaa["span_f1"] is not None:
            delta_iaa = latest_iaa["span_f1"] - 0.80
            st.metric(
                "Latest IAA Span F1",
                f"{latest_iaa['span_f1']:.4f}",
                f"{delta_iaa:+.4f} vs 0.80",
            )
        else:
            st.metric("Latest IAA Span F1", "Pending", "Target >= 0.80")
    with c4:
        if latest_qa:
            err = latest_qa["error_count"]
            st.metric("Label QA Errors", str(err), "PASS (0)" if err == 0 else "Fix required", delta_color="inverse")
        else:
            st.metric("Label QA Errors", "Pending", "Target = 0")
    with c5:
        if latest_snap:
            leak = "LEAK!" if latest_snap["has_doc_leakage"] else "0 Leakage"
            st.metric("Article Leakage", leak, f"Seed {latest_snap['seed']}")
        else:
            st.metric("Article Leakage", "Pending", "Target = 0")

    tab_score, tab_data, tab_qa, tab_train, tab_record, tab_team = st.tabs(
        [
            "Target Scorecard",
            "Dataset & Leakage",
            "IAA & QA Gate",
            "XLM-R Training Runs",
            "Upload / Record Results",
            "Team Progress Log",
        ]
    )

    # ------------------------------------------------------------------
    # Tab 1: Desired vs Actual Scorecard
    with tab_score:
        st.subheader("Expected vs Actual Quality Benchmarks")
        score_rows = [
            {
                "Stage": item["stage"],
                "Quality Benchmark": item["label"],
                "Expected": f"{item['operator']} {item['expected_value']:g}",
                "Actual": "Pending" if item["actual_value"] is None else f"{item['actual_value']:g}",
                "Gap": "—" if item["delta"] is None else f"{item['delta']:+g}",
                "Status": _status_badge(item["status"]),
                "Owner": item["owner"],
                "Notes": item["notes"],
            }
            for item in scorecard["items"]
        ]
        st.dataframe(pd.DataFrame(score_rows), hide_index=True)

        with st.expander("Edit Expected Quality Targets"):
            targets = db.list_targets()
            target_keys = [t["target_key"] for t in targets]
            selected_key = st.selectbox("Select Target Benchmark", target_keys)
            current = next(t for t in targets if t["target_key"] == selected_key)
            with st.form("edit_target_form"):
                col_a, col_b, col_c = st.columns(3)
                new_op = col_a.selectbox(
                    "Condition", [">=", "<=", "="], index=[">=", "<=", "="].index(current["operator"])
                )
                new_val = col_b.number_input("Expected Value", value=float(current["expected_value"]), format="%.4f")
                new_owner = col_c.text_input("Owner", value=current["owner"])
                new_notes = st.text_input("Notes", value=current["notes"])
                if st.form_submit_button("Save Target"):
                    db.upsert_target(
                        target_key=current["target_key"],
                        stage=current["stage"],
                        label=current["label"],
                        expected_value=new_val,
                        operator=new_op,
                        unit=current["unit"],
                        owner=new_owner,
                        notes=new_notes,
                    )
                    st.success(f"Updated target `{selected_key}`.")
                    st.rerun()

    # ------------------------------------------------------------------
    # Tab 2: Dataset & Zero-Leakage Funnel
    with tab_data:
        st.subheader("Dataset Lineage, Entity Balance & Zero-Leakage Audit")
        snapshots = db.list_snapshots()
        if not snapshots:
            st.info(
                "No dataset snapshots recorded yet. Run `checker/prepare_splits.py` and upload "
                "`split_report.json` in the **Upload / Record Results** tab or via `dashboard/sync.py split`."
            )
        else:
            snap = snapshots[0]
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**Latest Snapshot:** `{snap['snapshot_name']}` ({snap['sources']} | {snap['licence']})")
                funnel_df = pd.DataFrame(
                    {
                        "Stage": ["Raw Articles", "Cleaned", "Segmented", "Annotated", "Train", "Dev", "Test"],
                        "Count": [
                            snap["raw_articles"],
                            snap["cleaned_sentences"],
                            snap["segmented_sentences"],
                            snap["annotated_sentences"],
                            snap["train_sentences"],
                            snap["dev_sentences"],
                            snap["test_sentences"],
                        ],
                    }
                ).set_index("Stage")
                st.bar_chart(funnel_df)
            with col_right:
                st.markdown("**Entity Distribution (PER / LOC / ORG)**")
                ent_df = pd.DataFrame(
                    {
                        "Entity": ["PER", "LOC", "ORG"],
                        "Count": [snap["per_entities"], snap["loc_entities"], snap["org_entities"]],
                    }
                ).set_index("Entity")
                st.bar_chart(ent_df)
                if snap["has_doc_leakage"]:
                    st.error("Document leakage detected in latest snapshot! Re-run `checker/prepare_splits.py`.")
                else:
                    st.success(
                        f"Zero document leakage verified (seed={snap['seed']}, "
                        f"broken Khmer clusters={snap['broken_khmer_clusters']})."
                    )

            st.markdown("**All Recorded Snapshots**")
            st.dataframe(pd.DataFrame(snapshots), hide_index=True)

    # ------------------------------------------------------------------
    # Tab 3: IAA & Label QA Gate
    with tab_qa:
        st.subheader("Stage 4 Quality Gate — Inter-Annotator Agreement & Label Checks")
        iaa_list = db.list_qa_checks("iaa")
        qa_list = db.list_qa_checks("label_audit")

        col_iaa, col_qa = st.columns(2)
        with col_iaa:
            st.markdown("### Inter-Annotator Agreement (`checker/agreement.py`)")
            if not iaa_list:
                st.info("No IAA runs recorded yet. Target: Span F1 >= 0.80 on ~200 double-labeled sentences.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Batch": r["batch_name"],
                                "Annotators": r["annotators"],
                                "Sents": r["sentences_checked"],
                                "Span F1": r["span_f1"],
                                "Token Kappa": r["token_kappa"],
                                "Entity Kappa": r["entity_only_kappa"],
                                "Pass": r["passed"],
                                "Date": r["created_at"],
                            }
                            for r in iaa_list
                        ]
                    ),
                    hide_index=True,
                )

        with col_qa:
            st.markdown("### Automated Label Audit (`checker/label_checker.py`)")
            if not qa_list:
                st.info("No label audits recorded yet. Target: 0 errors before training.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Batch": r["batch_name"],
                                "Sents": r["sentences_checked"],
                                "Errors": r["error_count"],
                                "Warnings": r["warning_count"],
                                "Pass": r["passed"],
                                "By": r["recorded_by"],
                                "Date": r["created_at"],
                            }
                            for r in qa_list
                        ]
                    ),
                    hide_index=True,
                )

    # ------------------------------------------------------------------
    # Tab 4: XLM-R Training Runs
    with tab_train:
        st.subheader("Stage 5 Experiments — XLM-RoBERTa / mDeBERTa-v3 NER Runs")
        runs = db.list_training_runs()
        if not runs:
            st.info(
                "No training runs recorded yet. Once you finish a Colab GPU run, upload "
                "`run_summary.json` / `metrics.json` or run `python dashboard/sync.py train ...`."
            )
        else:
            table_rows = [
                {
                    "Run": r["run_name"],
                    "Model": r["model_name"],
                    "Dataset": r["snapshot_name"],
                    "Split": r["split_evaluated"],
                    "Epochs": r["epochs"],
                    "LR": r["learning_rate"],
                    "Prec": round(r["overall_precision"], 4),
                    "Rec": round(r["overall_recall"], 4),
                    "Overall F1": round(r["overall_f1"], 4),
                    "PER F1": round(r["per_f1"], 4),
                    "LOC F1": round(r["loc_f1"], 4),
                    "ORG F1": round(r["org_f1"], 4),
                    "Meets 0.70": r["meets_target"],
                    "Device": r["device"],
                    "Date": r["created_at"],
                }
                for r in runs
            ]
            st.dataframe(pd.DataFrame(table_rows), hide_index=True)

            st.download_button(
                "Download Experiment Table (CSV)",
                data=_to_csv_bytes(table_rows),
                file_name="khmerannotate_experiment_runs.csv",
                mime="text/csv",
            )

            run_names = [f"#{r['id']} — {r['run_name']} ({r['model_name']})" for r in runs]
            chosen_idx = st.selectbox("Inspect Run Breakdown", range(len(runs)), format_func=lambda i: run_names[i])
            chosen = runs[chosen_idx]

            col_f1, col_err = st.columns(2)
            with col_f1:
                st.markdown("**Per-Entity Span F1 vs Target (0.70)**")
                f1_df = pd.DataFrame(
                    {
                        "Metric": ["PER", "LOC", "ORG", "Overall", "Target"],
                        "F1": [
                            chosen["per_f1"],
                            chosen["loc_f1"],
                            chosen["org_f1"],
                            chosen["overall_f1"],
                            0.70,
                        ],
                    }
                ).set_index("Metric")
                st.bar_chart(f1_df)
            with col_err:
                st.markdown("**Span Error Breakdown**")
                err_df = pd.DataFrame(
                    {
                        "Category": ["Correct", "Boundary", "Type", "Boundary+Type", "Missed (FN)", "Spurious (FP)"],
                        "Spans": [
                            chosen["err_correct"],
                            chosen["err_boundary"],
                            chosen["err_type"],
                            chosen["err_boundary_type"],
                            chosen["err_missed"],
                            chosen["err_spurious"],
                        ],
                    }
                ).set_index("Category")
                st.bar_chart(err_df)

    # ------------------------------------------------------------------
    # Tab 5: Upload JSON Artifacts or Log Manually
    with tab_record:
        st.subheader("Import Pipeline JSON Artifact (Recommended)")
        st.caption(
            "Drop JSON files generated by `checker/` or `baseline/` directly into the database—no manual typing."
        )

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            artifact_type = st.selectbox(
                "Artifact Type",
                [
                    "Training Run (run_summary.json / metrics.json)",
                    "IAA Report (iaa_metrics.json)",
                    "Label QA Audit (qa_report.json)",
                    "Split & Lineage Report (split_report.json)",
                ],
            )
            recorded_by = st.selectbox("Recorded By", TEAM_MEMBERS)
            label_name = st.text_input("Run / Batch / Snapshot Name", value="v1-run-01")
            notes = st.text_input("Notes (optional)", value="")

        with col_up2:
            uploaded_main = st.file_uploader("Primary JSON File", type=["json"], key="main_json")
            uploaded_eval = None
            if artifact_type.startswith("Training Run"):
                uploaded_eval = st.file_uploader(
                    "Optional eval metrics.json (for span error counts)", type=["json"], key="eval_json"
                )
                snapshot_ref = st.text_input("Dataset Snapshot Name", value="v1-curated")
            elif artifact_type.startswith("Split"):
                src_text = st.text_input("Sources", value="VOA Khmer")
                lic_text = st.text_input("Licence", value="Public Domain")

        if st.button("Save Uploaded Artifact to Database", type="primary"):
            if uploaded_main is None:
                st.warning("Please upload a JSON file first.")
            else:
                try:
                    main_payload = json.loads(uploaded_main.getvalue().decode("utf-8"))
                    if artifact_type.startswith("Training Run"):
                        eval_payload = (
                            json.loads(uploaded_eval.getvalue().decode("utf-8"))
                            if uploaded_eval is not None
                            else None
                        )
                        if "metrics" in main_payload and eval_payload is None:
                            row_id = ingest_training_payload(
                                db,
                                run_name=label_name,
                                eval_payload=main_payload,
                                snapshot_name=snapshot_ref,
                                recorded_by=recorded_by,
                                notes=notes,
                            )
                        else:
                            row_id = ingest_training_payload(
                                db,
                                run_name=label_name,
                                summary_payload=main_payload,
                                eval_payload=eval_payload,
                                snapshot_name=snapshot_ref,
                                recorded_by=recorded_by,
                                notes=notes,
                            )
                    elif artifact_type.startswith("IAA"):
                        row_id = ingest_iaa_payload(
                            db, main_payload, batch_name=label_name, recorded_by=recorded_by, notes=notes
                        )
                    elif artifact_type.startswith("Label QA"):
                        row_id = ingest_qa_payload(
                            db, main_payload, batch_name=label_name, recorded_by=recorded_by, notes=notes
                        )
                    else:
                        row_id = ingest_split_payload(
                            db,
                            main_payload,
                            snapshot_name=label_name,
                            sources=src_text,
                            licence=lic_text,
                            recorded_by=recorded_by,
                            notes=notes,
                        )
                    st.success(f"Saved record #{row_id} to {db.backend_label}!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not ingest JSON: {exc}")

        st.divider()
        with st.expander("Manual Training Run Entry"):
            with st.form("manual_run_form"):
                mc1, mc2, mc3 = st.columns(3)
                m_run = mc1.text_input("Run Name", value="xlmr-baseline-v1")
                m_model = mc2.text_input("Model Name", value="xlm-roberta-base")
                m_snap = mc3.text_input("Dataset Snapshot", value="v1-curated")

                mc4, mc5, mc6, mc7 = st.columns(4)
                m_prec = mc4.number_input("Precision", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
                m_rec = mc5.number_input("Recall", min_value=0.0, max_value=1.0, value=0.74, step=0.01)
                m_f1 = mc6.number_input("Overall F1", min_value=0.0, max_value=1.0, value=0.745, step=0.01)
                m_split = mc7.selectbox("Split", ["test", "dev"])

                mc8, mc9, mc10 = st.columns(3)
                m_per = mc8.number_input("PER F1", min_value=0.0, max_value=1.0, value=0.78, step=0.01)
                m_loc = mc9.number_input("LOC F1", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
                m_org = mc10.number_input("ORG F1", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

                m_by = st.selectbox("Team Member", TEAM_MEMBERS, key="manual_by")
                m_notes = st.text_input("Notes", key="manual_notes")
                if st.form_submit_button("Record Training Run"):
                    row_id = db.record_training_run(
                        run_name=m_run,
                        snapshot_name=m_snap,
                        model_name=m_model,
                        split_evaluated=m_split,
                        overall_precision=m_prec,
                        overall_recall=m_rec,
                        overall_f1=m_f1,
                        macro_f1=round((m_per + m_loc + m_org) / 3.0, 4),
                        per_f1=m_per,
                        loc_f1=m_loc,
                        org_f1=m_org,
                        recorded_by=m_by,
                        notes=m_notes,
                    )
                    st.success(f"Saved manual training run #{row_id}!")
                    st.rerun()

    # ------------------------------------------------------------------
    # Tab 6: Team Progress Log
    with tab_team:
        st.subheader("Shared Team Milestone & Blocker Log")
        with st.form("milestone_form"):
            tc1, tc2, tc3 = st.columns(3)
            member = tc1.selectbox("Member", TEAM_MEMBERS)
            stage = tc2.selectbox("Stage", STAGES)
            status = tc3.selectbox("Status", ["in_progress", "done", "blocked", "planned"])
            summary = st.text_area("Update / Blocker Summary", height=80)
            if st.form_submit_button("Post Update"):
                if not summary.strip():
                    st.warning("Please enter a short summary.")
                else:
                    db.add_milestone_update(
                        member=member, stage=stage, status=status, summary=summary.strip()
                    )
                    st.success("Posted update to shared database.")
                    st.rerun()

        updates = db.list_milestone_updates()
        if not updates:
            st.info("No team updates posted yet.")
        else:
            st.dataframe(pd.DataFrame(updates), hide_index=True)


if __name__ == "__main__":
    main()
