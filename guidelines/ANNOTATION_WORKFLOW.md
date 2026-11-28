# Stage 3: Annotation Workflow (Local INCEpTION)

This folder contains a local Docker setup for **INCEpTION** with MariaDB
configured using `utf8mb4_bin` so Khmer Unicode bytes are stored without loss.

## 1. Export pre-segmented text batches (Stage 2b)

From the repository root:

```powershell
python cleaner/segment_words.py `
  --export-dir data/annotation_batches `
  --pilot-size 60 `
  --pool-size 2000 `
  --annotators 6 `
  --double-labeled 200
```

This creates paired `.txt` (for INCEpTION import) and `.jsonl` (provenance
metadata with `doc_id` and `url`) files in `data/annotation_batches/`:

- `smoke_5.txt` — 5 sentences to test login and whole-word snapping
- `pilot_60.txt` — 60 shared pilot sentences for all annotators
- `annotation_pool_*.txt` — main pool + `iaa_double_200.txt` + `annotator_1..6.txt`

## 2. Start INCEpTION locally

Make sure Docker Desktop is running, then from the repository root:

```powershell
docker compose -f guidelines/inception/docker-compose.yml up -d
docker compose -f guidelines/inception/docker-compose.yml logs -f inception
```

Wait until you see `Started INCEpTION`, then open **http://localhost:8080** and
sign in with `admin` / `admin` (change the password on first login).

To stop INCEpTION and free RAM (your database and annotations stay saved on
disk in `guidelines/inception/data` and `guidelines/inception/db_data`):

```powershell
docker compose -f guidelines/inception/docker-compose.yml down
```

## 3. Configure the INCEpTION project

1. **Create Project:** Name `KhmerAnnotate v1`, Mode `Annotation`.
2. **Create Tagset `KhmerNER`:** **Project -> Settings -> Tagsets -> Create**
   with exactly 3 tags: `PER`, `LOC`, `ORG`.
3. **Configure Named Entity Layer:** **Project -> Settings -> Layers -> Named entity**:
   - **Granularity:** `Tokens` *(snaps to whole Khmer words — never Characters)*
   - **Overlap mode:** `No overlap` *(guarantees valid flat BIO export)*
   - **Feature `value` -> Tagset:** `KhmerNER`
4. **Import Documents:** **Project -> Settings -> Documents -> Import**
   - **Format:** `Plain text (one sentence per line)` *(do not use plain text
     without line breaks, which merges Khmer lines into one sentence)*
   - Upload `smoke_5.txt` first, then `pilot_60.txt`, then the main batches.

## 4. Export to CoNLL-2002 and run Stage 4 QA

1. Export each annotator's files (for IAA) and the curated files as
   **`CoNLL-2002`** into `data/annotated/`.
2. Measure Inter-Annotator Agreement (`>= 0.80`):
   ```powershell
   python checker/agreement.py `
     --annotator-a data/annotated/ann1_iaa.conll `
     --annotator-b data/annotated/ann2_iaa.conll `
     --output-dir outputs/qa/iaa
   ```
3. Audit curated labels (`0` errors required):
   ```powershell
   python checker/label_checker.py `
     --input data/annotated/curated.conll `
     --output-dir outputs/qa/labels
   ```
4. Re-attach `doc_id` / `url` provenance and build leak-free 80/10/10 splits:
   ```powershell
   python checker/prepare_splits.py `
     --conll data/annotated/curated.conll `
     --metadata data/annotation_batches/annotation_pool_2000.jsonl `
     --output-dir data/curated
   ```
5. Hand off straight to Stage 5 (`baseline/train_xlmr.py`):
   ```powershell
   python baseline/train_xlmr.py `
     --train data/curated/train.conll `
     --dev data/curated/dev.conll `
     --test data/curated/test.conll `
     --dry-run
   ```
