"""Database layer for the KhmerAnnotate Quality & Training Tracker.

Supports two backends transparently:
1. **Neon PostgreSQL** (production / shared team deployment): activated
   whenever ``DATABASE_URL`` starts with ``postgres://`` or ``postgresql://``.
   Automatically enforces ``sslmode=require`` (required by Neon) and works
   with either ``psycopg`` (v3) or ``psycopg2``.
2. **Local SQLite** (zero-setup local fallback): activated when no Postgres
   ``DATABASE_URL`` is configured, storing data in ``data/dashboard_local.db``.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEFAULT_SQLITE_PATH = Path("data/dashboard_local.db")
SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"

DEFAULT_QUALITY_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "target_key": "broken_khmer_clusters",
        "stage": "Stage 2: Cleaner",
        "label": "Broken Khmer Character Clusters",
        "expected_value": 0.0,
        "operator": "=",
        "unit": "count",
        "owner": "Tools / Preprocessing",
        "notes": "No segmented token may start with coeng or a combining mark.",
    },
    {
        "target_key": "curated_sentences",
        "stage": "Stage 3: Annotation",
        "label": "Curated Annotated Sentences (Dataset v1)",
        "expected_value": 2000.0,
        "operator": ">=",
        "unit": "count",
        "owner": "All 6 Annotators",
        "notes": "Target Capstone I curated pool before 80/10/10 split.",
    },
    {
        "target_key": "iaa_span_f1",
        "stage": "Stage 4: Quality Gate",
        "label": "Inter-Annotator Agreement (Span F1)",
        "expected_value": 0.80,
        "operator": ">=",
        "unit": "score",
        "owner": "Rulebook & Curators",
        "notes": "Measured on ~200 double-labeled sentences via checker/agreement.py.",
    },
    {
        "target_key": "qa_open_errors",
        "stage": "Stage 4: Quality Gate",
        "label": "Label Checker Open Errors",
        "expected_value": 0.0,
        "operator": "=",
        "unit": "count",
        "owner": "Quality Gate",
        "notes": "BIO errors, broken clusters, or punctuation inside entity spans.",
    },
    {
        "target_key": "doc_leakage_count",
        "stage": "Stage 4: Quality Gate",
        "label": "Cross-Split Article Leakage",
        "expected_value": 0.0,
        "operator": "=",
        "unit": "count",
        "owner": "DBA / Provenance Gate",
        "notes": "Zero shared doc_id/url across train, dev, and test splits.",
    },
    {
        "target_key": "overall_f1",
        "stage": "Stage 5: XLM-R Model",
        "label": "Test Overall Span F1 (xlm-roberta-base)",
        "expected_value": 0.70,
        "operator": ">=",
        "unit": "score",
        "owner": "Model Lead (Sousdey)",
        "notes": "Primary Capstone I benchmark on held-out leak-free test split.",
    },
    {
        "target_key": "per_f1",
        "stage": "Stage 5: XLM-R Model",
        "label": "Test PER Span F1",
        "expected_value": 0.70,
        "operator": ">=",
        "unit": "score",
        "owner": "Model Lead (Sousdey)",
        "notes": "Person entity benchmark (titles like លោក excluded).",
    },
    {
        "target_key": "loc_f1",
        "stage": "Stage 5: XLM-R Model",
        "label": "Test LOC Span F1",
        "expected_value": 0.70,
        "operator": ">=",
        "unit": "score",
        "owner": "Model Lead (Sousdey)",
        "notes": "Location entity benchmark (ខេត្ត/ខណ្ឌ included, ប្រទេស/តំបន់ excluded).",
    },
    {
        "target_key": "org_f1",
        "stage": "Stage 5: XLM-R Model",
        "label": "Test ORG Span F1",
        "expected_value": 0.70,
        "operator": ">=",
        "unit": "score",
        "owner": "Model Lead (Sousdey)",
        "notes": "Organisation entity benchmark (outermost nested spans).",
    },
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_postgres_url(url: str) -> str:
    """Normalize ``postgres://`` to ``postgresql://`` and ensure ``sslmode=require`` for Neon."""
    cleaned = url.strip()
    if cleaned.startswith("postgres://"):
        cleaned = "postgresql://" + cleaned[len("postgres://") :]
    parts = urlsplit(cleaned)
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "sslmode" not in query_params:
        query_params["sslmode"] = "require"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment))


def is_postgres_url(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.strip().lower()
    return lowered.startswith("postgres://") or lowered.startswith("postgresql://")


def _to_sqlite_ddl(postgres_ddl: str) -> str:
    """Convert the Neon PostgreSQL schema DDL into equivalent SQLite DDL."""
    sql = postgres_ddl.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    sql = sql.replace("DOUBLE PRECISION", "REAL")
    sql = sql.replace("BOOLEAN NOT NULL DEFAULT FALSE", "INTEGER NOT NULL DEFAULT 0")
    sql = sql.replace("BOOLEAN NOT NULL DEFAULT TRUE", "INTEGER NOT NULL DEFAULT 1")
    return sql


def _adapt_placeholders(query: str, is_postgres: bool) -> str:
    """Translate ``?`` parameter placeholders to ``%s`` when targeting PostgreSQL."""
    return query.replace("?", "%s") if is_postgres else query


def evaluate_target(actual: float | None, expected: float, operator: str) -> str:
    """Return ``PASS``, ``FAIL``, or ``PENDING`` for a target comparison."""
    if actual is None:
        return "PENDING"
    if operator == ">=":
        return "PASS" if actual >= expected - 1e-9 else "FAIL"
    if operator == "<=":
        return "PASS" if actual <= expected + 1e-9 else "FAIL"
    if operator == "=":
        return "PASS" if abs(actual - expected) <= 1e-9 else "FAIL"
    raise ValueError(f"Unsupported operator: {operator!r}")


class QualityTrackerDB:
    """Dual-backend tracker database (Neon PostgreSQL or local SQLite)."""

    def __init__(self, database_url: str | None = None) -> None:
        raw_url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
        self.is_postgres = is_postgres_url(raw_url)
        if self.is_postgres:
            self.database_url = normalize_postgres_url(raw_url)
            self.sqlite_path: Path | None = None
        else:
            self.database_url = raw_url.strip() or str(DEFAULT_SQLITE_PATH)
            self.sqlite_path = Path(self.database_url)
            if str(self.sqlite_path) != ":memory:":
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def backend_label(self) -> str:
        return "Neon PostgreSQL" if self.is_postgres else f"Local SQLite ({self.sqlite_path})"

    def _connect_postgres(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row

            return psycopg.connect(self.database_url, row_factory=dict_row)
        except ImportError:
            pass
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL driver required for Neon DATABASE_URL. Install:\n"
                "  pip install -r dashboard/requirements-dashboard.txt"
            ) from exc

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        if self.is_postgres:
            conn = self._connect_postgres()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            assert self.sqlite_path is not None
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _execute(self, conn: Any, query: str, params: Sequence[Any] = ()) -> Any:
        adapted = _adapt_placeholders(query, self.is_postgres)
        cur = conn.cursor()
        cur.execute(adapted, tuple(params))
        return cur

    def _fetchall(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._connection() as conn:
            cur = self._execute(conn, query, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def _fetchone(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self._connection() as conn:
            cur = self._execute(conn, query, params)
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def init_db(self, *, seed_targets: bool = True) -> None:
        """Create tables/indexes if missing and seed default quality targets."""
        raw_ddl = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        ddl = raw_ddl if self.is_postgres else _to_sqlite_ddl(raw_ddl)
        statements = [stmt.strip() for stmt in ddl.split(";") if stmt.strip()]

        with self._connection() as conn:
            for statement in statements:
                self._execute(conn, statement)

        if seed_targets:
            now = utc_now_iso()
            existing = {row["target_key"] for row in self.list_targets()}
            for item in DEFAULT_QUALITY_TARGETS:
                if item["target_key"] not in existing:
                    self.upsert_target(
                        target_key=item["target_key"],
                        stage=item["stage"],
                        label=item["label"],
                        expected_value=item["expected_value"],
                        operator=item["operator"],
                        unit=item["unit"],
                        owner=item["owner"],
                        notes=item["notes"],
                        updated_at=now,
                    )

    # ------------------------------------------------------------------
    # Targets

    def list_targets(self) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT * FROM quality_targets ORDER BY stage ASC, target_key ASC"
        )

    def upsert_target(
        self,
        *,
        target_key: str,
        stage: str,
        label: str,
        expected_value: float,
        operator: str = ">=",
        unit: str = "score",
        owner: str = "Team",
        notes: str = "",
        updated_at: str | None = None,
    ) -> None:
        ts = updated_at or utc_now_iso()
        sql = """
            INSERT INTO quality_targets (
                target_key, stage, label, expected_value, operator, unit, owner, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (target_key) DO UPDATE SET
                stage = EXCLUDED.stage,
                label = EXCLUDED.label,
                expected_value = EXCLUDED.expected_value,
                operator = EXCLUDED.operator,
                unit = EXCLUDED.unit,
                owner = EXCLUDED.owner,
                notes = EXCLUDED.notes,
                updated_at = EXCLUDED.updated_at
        """
        with self._connection() as conn:
            self._execute(
                conn,
                sql,
                (target_key, stage, label, float(expected_value), operator, unit, owner, notes, ts),
            )

    # ------------------------------------------------------------------
    # Dataset snapshots

    def record_snapshot(
        self,
        *,
        snapshot_name: str,
        sources: str = "VOA Khmer",
        licence: str = "Public Domain",
        raw_articles: int = 0,
        cleaned_sentences: int = 0,
        segmented_sentences: int = 0,
        annotated_sentences: int = 0,
        train_sentences: int = 0,
        dev_sentences: int = 0,
        test_sentences: int = 0,
        total_entities: int = 0,
        per_entities: int = 0,
        loc_entities: int = 0,
        org_entities: int = 0,
        broken_khmer_clusters: int = 0,
        has_doc_leakage: bool = False,
        seed: int = 42,
        recorded_by: str = "team",
        notes: str = "",
        created_at: str | None = None,
    ) -> int:
        ts = created_at or utc_now_iso()
        leak_val: Any = bool(has_doc_leakage) if self.is_postgres else int(bool(has_doc_leakage))
        sql = """
            INSERT INTO dataset_snapshots (
                snapshot_name, sources, licence,
                raw_articles, cleaned_sentences, segmented_sentences, annotated_sentences,
                train_sentences, dev_sentences, test_sentences,
                total_entities, per_entities, loc_entities, org_entities,
                broken_khmer_clusters, has_doc_leakage, seed, recorded_by, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            snapshot_name, sources, licence,
            int(raw_articles), int(cleaned_sentences), int(segmented_sentences), int(annotated_sentences),
            int(train_sentences), int(dev_sentences), int(test_sentences),
            int(total_entities), int(per_entities), int(loc_entities), int(org_entities),
            int(broken_khmer_clusters), leak_val, int(seed), recorded_by, notes, ts,
        )
        return self._insert_returning_id(sql, params)

    def list_snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM dataset_snapshots ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            row["has_doc_leakage"] = bool(row["has_doc_leakage"])
        return rows

    # ------------------------------------------------------------------
    # QA checks (IAA & Label Checker)

    def record_qa_check(
        self,
        *,
        check_type: str,
        batch_name: str,
        annotators: str = "",
        sentences_checked: int = 0,
        span_f1: float | None = None,
        token_kappa: float | None = None,
        entity_only_kappa: float | None = None,
        error_count: int = 0,
        warning_count: int = 0,
        passed: bool = False,
        recorded_by: str = "team",
        notes: str = "",
        created_at: str | None = None,
    ) -> int:
        ts = created_at or utc_now_iso()
        passed_val: Any = bool(passed) if self.is_postgres else int(bool(passed))
        sql = """
            INSERT INTO qa_checks (
                check_type, batch_name, annotators, sentences_checked,
                span_f1, token_kappa, entity_only_kappa,
                error_count, warning_count, passed, recorded_by, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            check_type, batch_name, annotators, int(sentences_checked),
            span_f1, token_kappa, entity_only_kappa,
            int(error_count), int(warning_count), passed_val, recorded_by, notes, ts,
        )
        return self._insert_returning_id(sql, params)

    def list_qa_checks(self, check_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if check_type:
            rows = self._fetchall(
                "SELECT * FROM qa_checks WHERE check_type = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (check_type, limit),
            )
        else:
            rows = self._fetchall(
                "SELECT * FROM qa_checks ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        for row in rows:
            row["passed"] = bool(row["passed"])
        return rows

    # ------------------------------------------------------------------
    # Training runs

    def record_training_run(
        self,
        *,
        run_name: str,
        snapshot_name: str = "",
        model_name: str = "xlm-roberta-base",
        split_evaluated: str = "test",
        seed: int = 42,
        epochs: int = 8,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        max_length: int = 256,
        device: str = "T4 GPU",
        overall_precision: float = 0.0,
        overall_recall: float = 0.0,
        overall_f1: float = 0.0,
        macro_f1: float = 0.0,
        per_f1: float = 0.0,
        loc_f1: float = 0.0,
        org_f1: float = 0.0,
        err_correct: int = 0,
        err_boundary: int = 0,
        err_type: int = 0,
        err_boundary_type: int = 0,
        err_missed: int = 0,
        err_spurious: int = 0,
        target_f1: float = 0.70,
        recorded_by: str = "team",
        notes: str = "",
        created_at: str | None = None,
    ) -> int:
        ts = created_at or utc_now_iso()
        meets = float(overall_f1) >= float(target_f1)
        meets_val: Any = meets if self.is_postgres else int(meets)
        sql = """
            INSERT INTO training_runs (
                run_name, snapshot_name, model_name, split_evaluated,
                seed, epochs, batch_size, learning_rate, max_length, device,
                overall_precision, overall_recall, overall_f1, macro_f1,
                per_f1, loc_f1, org_f1,
                err_correct, err_boundary, err_type, err_boundary_type, err_missed, err_spurious,
                meets_target, recorded_by, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            run_name, snapshot_name, model_name, split_evaluated,
            int(seed), int(epochs), int(batch_size), float(learning_rate), int(max_length), device,
            float(overall_precision), float(overall_recall), float(overall_f1), float(macro_f1),
            float(per_f1), float(loc_f1), float(org_f1),
            int(err_correct), int(err_boundary), int(err_type),
            int(err_boundary_type), int(err_missed), int(err_spurious),
            meets_val, recorded_by, notes, ts,
        )
        return self._insert_returning_id(sql, params)

    def list_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM training_runs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            row["meets_target"] = bool(row["meets_target"])
        return rows

    # ------------------------------------------------------------------
    # Team milestone updates

    def add_milestone_update(
        self,
        *,
        member: str,
        stage: str,
        status: str,
        summary: str,
        created_at: str | None = None,
    ) -> int:
        ts = created_at or utc_now_iso()
        sql = """
            INSERT INTO milestone_updates (member, stage, status, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        return self._insert_returning_id(sql, (member, stage, status, summary, ts))

    def list_milestone_updates(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT * FROM milestone_updates ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )

    def _insert_returning_id(self, sql: str, params: Sequence[Any]) -> int:
        with self._connection() as conn:
            if self.is_postgres:
                cur = self._execute(conn, sql + " RETURNING id", params)
                row = cur.fetchone()
                return int(row["id"])
            cur = self._execute(conn, sql, params)
            return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Desired vs Actual Readiness Scorecard

    def compute_readiness_scorecard(self) -> dict[str, Any]:
        """Compare latest recorded metrics against every desired quality target."""
        targets = self.list_targets()
        snapshots = self.list_snapshots(limit=1)
        latest_snap = snapshots[0] if snapshots else None

        iaa_rows = self.list_qa_checks("iaa", limit=1)
        latest_iaa = iaa_rows[0] if iaa_rows else None

        qa_rows = self.list_qa_checks("label_audit", limit=1)
        latest_qa = qa_rows[0] if qa_rows else None

        runs = self.list_training_runs(limit=50)
        test_runs = [r for r in runs if r["split_evaluated"] == "test"]
        latest_run = test_runs[0] if test_runs else (runs[0] if runs else None)

        actual_map: dict[str, float | None] = {
            "broken_khmer_clusters": (
                float(latest_snap["broken_khmer_clusters"]) if latest_snap else None
            ),
            "curated_sentences": (
                float(
                    latest_snap["annotated_sentences"]
                    or (
                        latest_snap["train_sentences"]
                        + latest_snap["dev_sentences"]
                        + latest_snap["test_sentences"]
                    )
                )
                if latest_snap
                and (
                    latest_snap["annotated_sentences"]
                    or latest_snap["train_sentences"]
                    or latest_snap["dev_sentences"]
                    or latest_snap["test_sentences"]
                )
                else None
            ),
            "iaa_span_f1": (
                float(latest_iaa["span_f1"])
                if latest_iaa and latest_iaa["span_f1"] is not None
                else None
            ),
            "qa_open_errors": float(latest_qa["error_count"]) if latest_qa else None,
            "doc_leakage_count": (
                (1.0 if latest_snap["has_doc_leakage"] else 0.0) if latest_snap else None
            ),
            "overall_f1": float(latest_run["overall_f1"]) if latest_run else None,
            "per_f1": float(latest_run["per_f1"]) if latest_run else None,
            "loc_f1": float(latest_run["loc_f1"]) if latest_run else None,
            "org_f1": float(latest_run["org_f1"]) if latest_run else None,
        }

        items: list[dict[str, Any]] = []
        passed = 0
        failed = 0
        pending = 0

        for target in targets:
            key = target["target_key"]
            expected = float(target["expected_value"])
            operator = target["operator"]
            actual = actual_map.get(key)
            status = evaluate_target(actual, expected, operator)
            if status == "PASS":
                passed += 1
            elif status == "FAIL":
                failed += 1
            else:
                pending += 1

            delta = round(actual - expected, 4) if actual is not None else None
            items.append(
                {
                    "target_key": key,
                    "stage": target["stage"],
                    "label": target["label"],
                    "operator": operator,
                    "expected_value": expected,
                    "actual_value": round(actual, 4) if actual is not None else None,
                    "delta": delta,
                    "unit": target["unit"],
                    "owner": target["owner"],
                    "status": status,
                    "notes": target["notes"],
                }
            )

        total = len(items)
        readiness_pct = round((passed / total) * 100.0, 1) if total else 0.0

        return {
            "total_targets": total,
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "readiness_pct": readiness_pct,
            "latest_snapshot": latest_snap,
            "latest_iaa": latest_iaa,
            "latest_qa": latest_qa,
            "latest_run": latest_run,
            "items": items,
        }
