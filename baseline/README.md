# Baseline XLM-RoBERTa NER (`baseline/`)

Training, evaluation, error analysis, and prediction scripts for Khmer NER
with flat whole-word BIO labels (`O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`,
`B-ORG`, `I-ORG`).

## Expected input format (CoNLL-2002)

One word and one BIO tag per line (tab or space separated), with a blank line
between sentences. Optional `# doc_id = ...` or `# url = ...` comments keep
sentences from the same article together during splitting so there is no
train/test leakage.

```text
# doc_id = voa-12345
# url = https://khmer.voanews.com/a/example/12345.html
លោក	O
សុខ	B-PER
ដារ៉ា	I-PER
នៅ	O
ភ្នំពេញ	B-LOC
។	O
```

Pre-tokenized `.jsonl` files (`{"tokens": [...], "tags": [...], "doc_id": "..."}`)
are also accepted.

## 1. Validate data locally (no GPU or PyTorch required)

Before uploading curated data to a GPU machine, validate BIO tags and check
for document leakage:

```bash
python baseline/train_xlmr.py \
  --train data/curated/train.conll \
  --dev data/curated/dev.conll \
  --test data/curated/test.conll \
  --dry-run
```

If you have one combined annotated file instead of pre-split files, pass
`--input` to create leak-free 80/10/10 document splits automatically:

```bash
python baseline/train_xlmr.py \
  --input data/annotated/all_curated.conll \
  --output-dir outputs/xlmr_baseline \
  --dry-run
```

## 2. Train on a GPU (recommended: Google Colab free T4)

`xlm-roberta-base` has 278M parameters and is much faster and safer to train
on a GPU than on an 8 GB CPU laptop.

```bash
# On Google Colab (with Runtime -> Change runtime type -> T4 GPU)
pip install -r baseline/requirements-baseline.txt

python baseline/train_xlmr.py \
  --train data/curated/train.conll \
  --dev data/curated/dev.conll \
  --test data/curated/test.conll \
  --output-dir outputs/xlmr_baseline \
  --epochs 8 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --seed 42
```

To compare against `microsoft/mdeberta-v3-base`, change only `--model-name`:

```bash
python baseline/train_xlmr.py \
  --model-name microsoft/mdeberta-v3-base \
  --train data/curated/train.conll \
  --dev data/curated/dev.conll \
  --test data/curated/test.conll \
  --output-dir outputs/mdeberta_baseline
```

Training saves:

- `outputs/xlmr_baseline/best_model/` — best checkpoint by dev span F1
- `outputs/xlmr_baseline/training_history.json` — per-epoch train/dev loss & F1
- `outputs/xlmr_baseline/run_summary.json` — exact config, counts, leakage & metrics
- `outputs/xlmr_baseline/dev_predictions.conll` and `test_predictions.conll`
- `outputs/xlmr_baseline/dev_error_report.html` and `test_error_report.html`

## 3. Evaluate & generate 300-DPI charts

From a saved checkpoint:

```bash
python baseline/evaluate.py \
  --model-dir outputs/xlmr_baseline/best_model \
  --data data/curated/test.conll \
  --history outputs/xlmr_baseline/training_history.json \
  --output-dir outputs/xlmr_baseline/eval
```

Or locally without PyTorch/GPU from a saved 3-column `test_predictions.conll`:

```bash
python baseline/evaluate.py \
  --predictions outputs/xlmr_baseline/test_predictions.conll \
  --history outputs/xlmr_baseline/training_history.json \
  --output-dir outputs/xlmr_baseline/eval
```

Outputs `metrics.json`, `error_report.html` (open in a browser to read Khmer
spans properly), and 4 PNG charts (`per_entity_prf1.png`,
`entity_confusion_matrix.png`, `error_breakdown.png`, `training_curves.png`).

## 4. Tag new Khmer sentences (`predict.py`)

```bash
python baseline/predict.py \
  --model-dir outputs/xlmr_baseline/best_model \
  --input data/cleaned/unlabeled_segmented.txt \
  --output outputs/predictions.jsonl
```

JSONL output includes extracted entity spans plus `least_confidence` and
`mean_entropy` per sentence for Active Learning selection.

## Tests

```bash
python -m unittest discover -s tests -t .
```
