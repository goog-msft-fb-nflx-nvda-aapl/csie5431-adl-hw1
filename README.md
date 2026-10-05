# ADL HW1 (A1) — User State Prediction

Bilingual (zh/en) multi-label sales-intent classification. Architecture: a **language-routed ensemble** — a separate fine-tuned encoder per language, routed by the `language` field already present in `test.json`/`train.jsonl` (no detection needed), rather than one bilingual model. Every route ignores dialogue context entirely (utterance-only input) — found empirically to outperform every context-inclusion strategy tried, on both languages, at every training-data scale.

**Status**: final. Submitted model is Candidate C (BGE-M3; zh single checkpoint + en 2-way ensemble) — see "Model performance" below. `docs/TODO.md`/`docs/WORKLOG.md` have the full experiment log (~71 runs) including the two earlier, superseded candidates (A, B) kept for the report's "Model variants" discussion. `docs/REPRODUCE.md` is the consolidated command reference for all three candidates plus the auxiliary experiments. Course spec and lecture reference material are under `docs/spec/`; the initial literature/model survey is under `docs/survey/`.

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

Full commands for all three candidates, plus the auxiliary (negative-result) experiments, are in **`docs/REPRODUCE.md`**. Short version, all commands run from `code/scripts/`:

**1. Split train/dev** (stratified by label + language, no external dependency, deterministic):
```bash
python split_data.py --train_path ../../data/train.jsonl --out_train ../../data/train_split.jsonl --out_dev ../../data/dev_split.jsonl
```

**2. Train + tune + diagnostic-check each route** — see `docs/REPRODUCE.md` sections 1–3 for the exact `train.py`/`threshold_tune.py`/`route_combine_eval.py` invocations for each of the three candidates (simple single-checkpoint, ensembled+LR-tuned, and BGE-M3).

**3. Package for offline inference** (bundles full model+tokenizer via `save_pretrained`, not just weights, so `run.sh` never needs network access; `--fp16` halves the package size — needed for the BGE-M3 candidate to fit the 4GB download budget):
```bash
python package_final.py --run_name <run_name> --out_dir ../../models/zh [--fp16]
python package_final.py --run_name <run_name> --out_dir ../../models/en [--fp16]
```
`package_final.py` also supports packaging a multi-checkpoint ensemble for one route: pass multiple `--run_name` values (e.g. the final en route is `--run_name N1_bgem3_en_seqft_zh N2_bgem3_en_seqft_seed1 --out_dir ../models/en --fp16`). It re-tunes thresholds fresh from each member's own saved dev logits and writes an `ensemble_meta.json`; `predict_final.py`/`run.sh` detect this automatically and average sigmoid probabilities across members at inference time.

**4. Archive and upload** `models/` (as `models.tar.gz`) to Google Drive, and put the file's share-id into `download.sh`'s `GDRIVE_FILE_ID` variable.

## Running inference

```bash
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv
```

`run.sh` calls `code/scripts/predict_final.py`, which loads both routes from `models/zh` and `models/en` (via `local_files_only=True` — no network calls), routes each test example by its `language` field, and writes `prediction.csv` in the required format (verified byte-identical header to `sample_prediction.csv`, `\n` line endings).

## Model performance (final submitted = Candidate C; A/B kept for comparison)

| Candidate | Public test (Macro/Micro) | Dev split (Macro/Micro) |
|---|---|---|
| A — simple, single checkpoint/route | 0.8057 / 0.8238 | 0.8502 / 0.8556 |
| B — ensembled + LR-tuned | 0.8218 / 0.8316 | 0.8335 / 0.8442 |
| C — BGE-M3, zh single checkpoint + en 2-way zh→en-sequential-FT ensemble (final, packaged 2026-09-25) | **0.8396 / 0.8538** | **0.8778 / 0.8781** |

Challenge grading tier is Macro-F1 ≥ 0.82, Micro-F1 ≥ 0.83 — **Candidate C clears it with real margin on both metrics, on both the dev split and the public-test diagnostic**, with a clean (non-public-test-driven) selection methodology. The en-route ensemble came from a multi-seed reliability check (round-2 Deep Research survey flagged the small en dev split as unreliable for single-run comparisons — confirmed on our own data, then fixed via seed-ensembling rather than just noted). It uses 2 of the 4 en checkpoints trained during the round-2 backlog, chosen as the best-performing pair that fits the assignment's 4GB download budget (the full 4-way ensemble doesn't fit). `docs/WORKLOG.md` has the full session log and `docs/REPRODUCE.md` the consolidated reproduction commands, including the exact `package_final.py` invocation for the final submission.
