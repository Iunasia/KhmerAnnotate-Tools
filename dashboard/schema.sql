-- KhmerAnnotate Quality & Training Tracker Schema (Neon PostgreSQL)
-- Run automatically by dashboard/db.py on startup, or paste into the Neon SQL Editor.

CREATE TABLE IF NOT EXISTS quality_targets (
    target_key      TEXT PRIMARY KEY,
    stage           TEXT NOT NULL,
    label           TEXT NOT NULL,
    expected_value  DOUBLE PRECISION NOT NULL,
    operator        TEXT NOT NULL CHECK (operator IN ('>=', '<=', '=')),
    unit            TEXT NOT NULL DEFAULT 'score',
    owner           TEXT NOT NULL DEFAULT 'Team',
    notes           TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id                      SERIAL PRIMARY KEY,
    snapshot_name           TEXT NOT NULL,
    sources                 TEXT NOT NULL DEFAULT 'VOA Khmer',
    licence                 TEXT NOT NULL DEFAULT 'Public Domain',
    raw_articles            INTEGER NOT NULL DEFAULT 0,
    cleaned_sentences       INTEGER NOT NULL DEFAULT 0,
    segmented_sentences     INTEGER NOT NULL DEFAULT 0,
    annotated_sentences     INTEGER NOT NULL DEFAULT 0,
    train_sentences         INTEGER NOT NULL DEFAULT 0,
    dev_sentences           INTEGER NOT NULL DEFAULT 0,
    test_sentences          INTEGER NOT NULL DEFAULT 0,
    total_entities          INTEGER NOT NULL DEFAULT 0,
    per_entities            INTEGER NOT NULL DEFAULT 0,
    loc_entities            INTEGER NOT NULL DEFAULT 0,
    org_entities            INTEGER NOT NULL DEFAULT 0,
    broken_khmer_clusters   INTEGER NOT NULL DEFAULT 0,
    has_doc_leakage         BOOLEAN NOT NULL DEFAULT FALSE,
    seed                    INTEGER NOT NULL DEFAULT 42,
    recorded_by             TEXT NOT NULL DEFAULT 'team',
    notes                   TEXT NOT NULL DEFAULT '',
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_checks (
    id                  SERIAL PRIMARY KEY,
    check_type          TEXT NOT NULL CHECK (check_type IN ('iaa', 'label_audit')),
    batch_name          TEXT NOT NULL,
    annotators          TEXT NOT NULL DEFAULT '',
    sentences_checked   INTEGER NOT NULL DEFAULT 0,
    span_f1             DOUBLE PRECISION,
    token_kappa         DOUBLE PRECISION,
    entity_only_kappa   DOUBLE PRECISION,
    error_count         INTEGER NOT NULL DEFAULT 0,
    warning_count       INTEGER NOT NULL DEFAULT 0,
    passed              BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_by         TEXT NOT NULL DEFAULT 'team',
    notes               TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_runs (
    id                  SERIAL PRIMARY KEY,
    run_name            TEXT NOT NULL,
    snapshot_name       TEXT NOT NULL DEFAULT '',
    model_name          TEXT NOT NULL DEFAULT 'xlm-roberta-base',
    split_evaluated     TEXT NOT NULL DEFAULT 'test' CHECK (split_evaluated IN ('dev', 'test')),
    seed                INTEGER NOT NULL DEFAULT 42,
    epochs              INTEGER NOT NULL DEFAULT 8,
    batch_size          INTEGER NOT NULL DEFAULT 16,
    learning_rate       DOUBLE PRECISION NOT NULL DEFAULT 0.00003,
    max_length          INTEGER NOT NULL DEFAULT 256,
    device              TEXT NOT NULL DEFAULT 'T4 GPU',
    overall_precision   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    overall_recall      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    overall_f1          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    macro_f1            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    per_f1              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    loc_f1              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    org_f1              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    err_correct         INTEGER NOT NULL DEFAULT 0,
    err_boundary        INTEGER NOT NULL DEFAULT 0,
    err_type            INTEGER NOT NULL DEFAULT 0,
    err_boundary_type   INTEGER NOT NULL DEFAULT 0,
    err_missed          INTEGER NOT NULL DEFAULT 0,
    err_spurious        INTEGER NOT NULL DEFAULT 0,
    meets_target        BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_by         TEXT NOT NULL DEFAULT 'team',
    notes               TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS milestone_updates (
    id          SERIAL PRIMARY KEY,
    member      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('planned', 'in_progress', 'blocked', 'done')),
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_created ON dataset_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_checks_type_created ON qa_checks (check_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_milestones_created ON milestone_updates (created_at DESC);
