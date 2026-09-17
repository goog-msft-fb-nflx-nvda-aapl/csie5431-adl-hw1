# ADL HW1 (A1) — User State Prediction

Bilingual (zh/en) multi-label sales-intent classification, submitted as a **language-routed ensemble**:
- Chinese examples → `hfl/chinese-macbert-base`, fine-tuned on the zh subset of `train.jsonl`
- English examples → `roberta-base`, fine-tuned on the en subset of `train.jsonl`
- Routing uses the `language` field already present in `test.json` — no language detection needed.
- Each route ignores dialogue context entirely (utterance-only input) — this was found empirically to outperform every context-inclusion strategy tried, on both languages. See `report.pdf` / `WORKLOG.md` for the full ablation evidence.

Full experiment log with every configuration tried and its measured result is in `WORKLOG.md`; a running to-do/status file is in `TODO.md`.

## Environment

```bash
conda create -n adl_hw1_env python=3.10
conda activate adl_hw1_env
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126   # or the cuXXX index matching your driver
pip install transformers==4.50.0 datasets==2.21.0 accelerate==0.34.2 scikit-learn==1.7.2 evaluate matplotlib gdown pandas tqdm sentencepiece protobuf numpy
```

(Note: `scikit-learn==1.9.0` as listed in the assignment's allowed-package list requires Python ≥3.11 and cannot install under Python 3.10 — see `TA_QUESTIONS.md` item 8. We used `1.7.2`, the latest version compatible with Python 3.10, for our own dev tooling; it is not used anywhere in `run.sh`.)

Data (`train.jsonl`, `context.json`, `test.json`, `public_test_gold.csv`) is expected under `data/` — not included in this submission per the assignment rules; place the provided files there before running the steps below.

## Reproducing training from scratch

All commands run from `code/`.

**1. Split train/dev** (stratified by label + language, no external dependency):
```bash
python split_data.py --train_path ../data/train.jsonl --out_train ../data/train_split.jsonl --out_dev ../data/dev_split.jsonl
```

**2. Train the zh route** (MacBERT, context dropped, weighted BCE, 20 epochs):
```bash
python train.py --model_key macbert --context_mode none --loss weighted_bce --lang_subset zh --epochs 20 --run_name zh_final
```

**3. Train the en route** (RoBERTa, context dropped, Asymmetric Loss, 20 epochs):
```bash
python train.py --model_key roberta --context_mode none --loss asl --lang_subset en --epochs 20 --run_name en_final
```

**4. Tune per-class decision thresholds for each route** (on its own dev split only — never on `public_test_gold.csv`):
```bash
python threshold_tune.py --run_name zh_final
python threshold_tune.py --run_name en_final
```

**5. (Optional) Diagnostic check against the public test set** — for our own sanity checking only, never used for model/threshold selection:
```bash
python predict_and_eval.py --run_name zh_final
python predict_and_eval.py --run_name en_final
python route_combine_eval.py --zh_run zh_final --en_run en_final
```

**6. Package both routes for offline inference** (bundles full model+tokenizer via `save_pretrained`, not just weights, so `run.sh` never needs network access):
```bash
python package_final.py --run_name zh_final --out_dir ../models/zh
python package_final.py --run_name en_final --out_dir ../models/en
```

**7. Archive and upload** `models/` (as `models.tar.gz`) to Google Drive, and put the file's share-id into `download.sh`'s `GDRIVE_FILE_ID` variable.

## Running inference

```bash
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv
```

`run.sh` calls `code/predict_final.py`, which loads both routes from `models/zh` and `models/en` (via `local_files_only=True` — no network calls), routes each test example by its `language` field, and writes `prediction.csv` in the required format (verified byte-identical header to `sample_prediction.csv`, `\n` line endings).

## Model performance (this session's final result)

| Split | Macro-F1 | Micro-F1 |
|---|---|---|
| Public test (500 ex., diagnostic — thresholds/model chosen on dev only) | 0.8057 | 0.8238 |
| Dev split (450 ex., stratified from train) | 0.8502 | 0.8556 |

See `WORKLOG.md` for the full session log (every model/context/loss ablation tried, with configs and reproduce commands) and `TODO.md` for open items and report-question mapping.
