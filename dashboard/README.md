# KhmerAnnotate Quality & Training Dashboard (`dashboard/`)

A shared **Streamlit + Neon PostgreSQL** dashboard that tracks desired Capstone
quality benchmarks against actual dataset lineage, IAA agreement, label QA
audits, zero-leakage split checks, and XLM-R NER training runs.

When `DATABASE_URL` is not set, it automatically falls back to a local SQLite
database (`data/dashboard_local.db`) so anyone can run it locally with zero
setup.

---

## 1. Deploy with Neon PostgreSQL + Streamlit Community Cloud (5 min, Free)

### Step A: Create your Neon PostgreSQL database
1. Sign in at [console.neon.tech](https://console.neon.tech) and create a
   project (e.g. `khmerannotate-tracker`).
2. Copy the **Connection String** (Pooled or Direct), e.g.:
   ```text
   postgresql://neondb_owner:...@ep-xxxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
   ```
3. *(Optional)* You can paste `dashboard/schema.sql` into the Neon SQL Editor
   to inspect/create the tables upfront — or skip this step because
   `dashboard/app.py` and `dashboard/sync.py` auto-create all tables and seed
   the 9 default Capstone quality targets on first startup.

### Step B: Deploy `dashboard/app.py` for your team
1. Push your branch (`feat/pipeline-stage2-to-4`, or merge into `main`) to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io) -> **Create app**:
   - **Repository:** `Iunasia/KhmerAnnotate-Tools`
   - **Branch:** `feat/pipeline-stage2-to-4` (or `main` once merged)
   - **Main file path:** `dashboard/app.py`
3. Open **Advanced settings -> Secrets** and paste:
   ```toml
   DATABASE_URL = "postgresql://neondb_owner:...@ep-xxxx.aws.neon.tech/neondb?sslmode=require"
   ```
4. Click **Deploy**. Share the live URL with all 6 teammates and your advisors.

> Never commit your real `DATABASE_URL` password to Git. Keep it in `.env`
> (git-ignored) or Streamlit Secrets.

---

## 2. Run Locally

```powershell
pip install -r dashboard/requirements.txt

# Local SQLite mode (no Neon needed)
streamlit run dashboard/app.py

# Connected directly to your Neon PostgreSQL database
$env:DATABASE_URL = "postgresql://...neon.tech/neondb?sslmode=require"
streamlit run dashboard/app.py
```

---

## 3. Auto-Sync Results from Local Runs or Google Colab to Neon

Instead of typing numbers manually, push the JSON reports produced by
`checker/` and `baseline/` straight into your live Neon database:

```bash
# Install Postgres driver on Colab or locally (lightweight, ~3 MB)
pip install -q "psycopg[binary]>=3.2"

# Point to your Neon database
export DATABASE_URL="postgresql://...neon.tech/neondb?sslmode=require"

# 1. Check live Expected vs Actual readiness scorecard in terminal
python dashboard/sync.py scorecard

# 2. Push Stage 4 IAA metrics (checker/agreement.py)
python dashboard/sync.py iaa \
  --file outputs/qa/iaa/iaa_metrics.json \
  --batch iaa_double_200 \
  --by "Sousdey (Model Lead)"

# 3. Push Stage 4 Label QA Gate (checker/label_checker.py)
python dashboard/sync.py qa \
  --file outputs/qa/labels/qa_report.json \
  --batch curated_v1 \
  --by "Sousdey (Model Lead)"

# 4. Push Stage 4 Leak-Free Split & Lineage Report (checker/prepare_splits.py)
python dashboard/sync.py split \
  --file data/curated/split_report.json \
  --name v1-curated \
  --sources "VOA Khmer" \
  --licence "Public Domain" \
  --raw-articles 200 \
  --cleaned-sentences 4867 \
  --by "Sousdey (Model Lead)"

# 5. Push Stage 5 XLM-R Training + Test Evaluation straight from Google Colab!
python dashboard/sync.py train \
  --summary outputs/xlmr_baseline/run_summary.json \
  --eval outputs/xlmr_baseline/eval/metrics.json \
  --run-name xlmr-v1-seed42 \
  --snapshot v1-curated \
  --by "Sousdey (Model Lead)"
```

Teammates can also upload those exact JSON files through the **Upload / Record
Results** tab in the deployed web dashboard.
