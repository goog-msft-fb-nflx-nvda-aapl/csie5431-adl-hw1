# ADL HW1 (A1) — User State Prediction

Bilingual (zh/en) multi-label sales-intent classification. Architecture (all candidates, final choice not yet made — see below): a **language-routed ensemble** — a separate fine-tuned encoder per language, routed by the `language` field already present in `test.json`/`train.jsonl` (no detection needed), rather than one bilingual model. Every route ignores dialogue context entirely (utterance-only input) — found empirically to outperform every context-inclusion strategy tried, on both languages, at every training-data scale.

**Status**: three candidate configurations exist, not yet resolved to one final submission — see `docs/TODO.md` for the current state and `docs/WORKLOG.md` for the full experiment log (every model/context/loss/LR variant tried and its measured result, ~45 runs). `docs/REPRODUCE.md` is the consolidated command reference for reproducing all three candidates plus the auxiliary experiments. Course spec and lecture reference material are under `docs/spec/`; the initial literature/model survey is under `docs/survey/`.

## Environment

```bash
conda create -n adl_hw1_env python=3.10
conda activate adl_hw1_env
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126   # or the cuXXX index matching your driver
pip install transformers==4.50.0 datasets==2.21.0 accelerate==0.34.2 scikit-learn==1.7.2 evaluate matplotlib gdown pandas tqdm sentencepiece protobuf numpy
```

(Note: `scikit-learn==1.9.0` as listed in the assignment's allowed-package list requires Python ≥3.11 and cannot install under Python 3.10 — see `docs/TA_QUESTIONS.md` item 8. We used `1.7.2`, the latest version compatible with Python 3.10, for our own dev tooling; it is not used anywhere in `run.sh`.)

Data (`train.jsonl`, `context.json`, `test.json`, `public_test_gold.csv`) is expected under `data/` — not included in this submission per the assignment rules; place the provided files there before running the steps below.

## Reproducing training from scratch

Full commands for all three candidates, plus the auxiliary (negative-result) experiments, are in **`docs/REPRODUCE.md`**. Short version, all commands run from `code/`:

**1. Split train/dev** (stratified by label + language, no external dependency, deterministic):
```bash
python split_data.py --train_path ../data/train.jsonl --out_train ../data/train_split.jsonl --out_dev ../data/dev_split.jsonl
```

**2. Train + tune + diagnostic-check each route** — see `docs/REPRODUCE.md` sections 1–3 for the exact `train.py`/`threshold_tune.py`/`route_combine_eval.py` invocations for each of the three candidates (simple single-checkpoint, ensembled+LR-tuned, and BGE-M3).

**3. Package for offline inference** (bundles full model+tokenizer via `save_pretrained`, not just weights, so `run.sh` never needs network access; `--fp16` halves the package size — needed for the BGE-M3 candidate to fit the 4GB download budget):
```bash
python package_final.py --run_name <run_name> --out_dir ../models/zh [--fp16]
python package_final.py --run_name <run_name> --out_dir ../models/en [--fp16]
```
Note: only single-checkpoint routes are packageable this way today — the ensembled candidate doesn't have `run.sh` support yet (see `docs/TODO.md`).

**4. Archive and upload** `models/` (as `models.tar.gz`) to Google Drive, and put the file's share-id into `download.sh`'s `GDRIVE_FILE_ID` variable.

## Running inference

```bash
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv
```

`run.sh` calls `code/predict_final.py`, which loads both routes from `models/zh` and `models/en` (via `local_files_only=True` — no network calls), routes each test example by its `language` field, and writes `prediction.csv` in the required format (verified byte-identical header to `sample_prediction.csv`, `\n` line endings).

## Model performance (three candidates, final choice not yet made)

| Candidate | Public test (Macro/Micro) | Dev split (Macro/Micro) |
|---|---|---|
| A — simple, single checkpoint/route | 0.8057 / 0.8238 | 0.8502 / 0.8556 |
| B — ensembled + LR-tuned | 0.8218 / 0.8316 | 0.8335 / 0.8442 |
| C — BGE-M3, zh single checkpoint + en 4-seed zh→en-sequential-FT ensemble (2026-09-23) | **0.8405 / 0.8527** | **0.8756 / 0.8760** |

Challenge grading tier is Macro-F1 ≥ 0.82, Micro-F1 ≥ 0.83 — **Candidate C clears it with real margin on both metrics, on both the dev split and the public-test diagnostic**, with a clean (non-public-test-driven) selection methodology. The en-route ensemble came from a multi-seed reliability check (round-2 Deep Research survey flagged the small en dev split as unreliable for single-run comparisons — confirmed on our own data, then fixed via seed-ensembling rather than just noted). See `docs/TODO.md` for the active research backlog and remaining open questions (whether a 568M-param retrieval model is in scope; B's selection-methodology caveat; en-route ensemble isn't packageable for `run.sh` yet). `docs/WORKLOG.md` has the full session log and `docs/REPRODUCE.md` the consolidated reproduction commands.
