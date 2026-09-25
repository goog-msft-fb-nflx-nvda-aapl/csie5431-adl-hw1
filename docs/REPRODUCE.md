# Reproduction guide

Single reference for reproducing every result this project has produced. All commands run from `code/` unless noted, and assume the environment from the root `README.md` is set up and the provided data (`train.jsonl`, `context.json`, `test.json`, `public_test_gold.csv`) is placed under `../data/`.

For the full experimental trail (every model/context/loss/LR variant tried, ~65 runs, with what each one found) see `WORKLOG.md` — every run there has its own `Reproduce:` command inline. This file only consolidates the commands that matter for reproducing the **live candidates** and the shared setup steps, so they don't have to be hunted down from a huge log.

## Final submitted candidate (2026-09-25)

zh: `F3_bgem3_zh` (single checkpoint, weighted BCE, lr=2e-5). en: **2-way ensemble** of `N1_bgem3_en_seqft_zh` + `N2_bgem3_en_seqft_seed1` (DB-Loss, lr=5e-5, warmup=0.15, warm-started from the zh checkpoint) — the best-performing pair out of the 4 en checkpoints trained during the round-2 backlog, chosen specifically because the full 4-way ensemble doesn't fit the 4GB download budget (see `WORKLOG.md`'s 2026-09-25 entry for the full 6-pair comparison). **Result: public test macro=0.8396/micro=0.8538, dev macro=0.8778/micro=0.8781.**

```bash
python package_final.py --run_name F3_bgem3_zh --out_dir ../models/zh --fp16
python package_final.py --run_name N1_bgem3_en_seqft_zh N2_bgem3_en_seqft_seed1 --out_dir ../models/en --fp16 --lang_subset en
```
(`--run_name` accepts multiple names for an ensembled route; see the two routes' individual `train.py` commands under Candidate C / the round-2 backlog sections of `WORKLOG.md` for how `F3`/`N1`/the seed runs were produced in the first place.)

## 0. Setup — always first

```bash
python split_data.py --train_path ../data/train.jsonl \
  --out_train ../data/train_split.jsonl --out_dev ../data/dev_split.jsonl
```
Stratified 85/15 train/dev split (2550/450) by random search over label + language rate deviation, seeded — deterministic given the same `train.jsonl`.

## 1. Candidate A — simplest, single checkpoint per route, selected purely on the dev split

| Route | Model | Config |
|---|---|---|
| zh | `hfl/chinese-macbert-base` | context=none, weighted BCE, 20 epochs |
| en | `roberta-base` | context=none, Asymmetric Loss, 20 epochs, lr=2e-5 |

```bash
python train.py --model_key macbert --context_mode none --loss weighted_bce \
  --lang_subset zh --epochs 20 --run_name C2b_macbert_none_wbce_zh_ep20
python train.py --model_key roberta --context_mode none --loss asl \
  --lang_subset en --epochs 20 --run_name C3b_roberta_none_asl_en_ep20

python threshold_tune.py --run_name C2b_macbert_none_wbce_zh_ep20
python threshold_tune.py --run_name C3b_roberta_none_asl_en_ep20

# diagnostic only — never used to pick the model/thresholds
python predict_and_eval.py --run_name C2b_macbert_none_wbce_zh_ep20
python predict_and_eval.py --run_name C3b_roberta_none_asl_en_ep20
python route_combine_eval.py --zh_run C2b_macbert_none_wbce_zh_ep20 --en_run C3b_roberta_none_asl_en_ep20
```
Result: public test macro=0.8057/micro=0.8238; dev macro=0.8502/micro=0.8556.

## 2. Candidate B — ensembled + learning-rate-tuned (methodology caveat: see `WORKLOG.md`'s "user pushback" section — selected partly via repeated public-test comparisons)

zh route needs one extra checkpoint (ASL instead of weighted BCE) on top of Candidate A's zh model:
```bash
python train.py --model_key macbert --context_mode none --loss asl \
  --lang_subset zh --epochs 10 --run_name C1_macbert_none_asl_zh
python threshold_tune.py --run_name C1_macbert_none_asl_zh
```

en route needs two extra checkpoints at different learning rates on top of Candidate A's en model:
```bash
python train.py --model_key roberta --context_mode none --loss asl --lr 5e-5 \
  --lang_subset en --epochs 20 --run_name D1_roberta_asl_en_lr5e-5
python train.py --model_key roberta --context_mode none --loss asl --lr 1e-4 \
  --lang_subset en --epochs 20 --run_name D1b_roberta_asl_en_lr1e-4
python threshold_tune.py --run_name D1_roberta_asl_en_lr5e-5
python threshold_tune.py --run_name D1b_roberta_asl_en_lr1e-4
```

Ensemble evaluation (averages sigmoid probabilities of the checkpoints, re-tunes thresholds on the averaged dev probabilities):
```bash
python ensemble_eval.py --runs C2b_macbert_none_wbce_zh_ep20 C1_macbert_none_asl_zh --lang zh
python ensemble_eval.py --runs C3b_roberta_none_asl_en_ep20 D1_roberta_asl_en_lr5e-5 D1b_roberta_asl_en_lr1e-4 --lang en
```
Result: public test macro=0.8218/micro=0.8316; dev macro=0.8335/micro=0.8442. (`ensemble_eval.py` writes its probabilities/thresholds to `/tmp/ensemble_<lang>_*` — there is no packaged `run.sh` support for a multi-checkpoint ensemble yet; only single-checkpoint routes are packageable via `package_final.py` today.)

## 3. Candidate C — `BAAI/bge-m3`, single checkpoint per route, no ensembling (current leading candidate, clears challenge tier on both dev and public test, cleanest selection methodology)

Note: an LR sweep (2026-09-23) found lr=5e-5 beats lr=1e-4 for the en route specifically (zh stayed at the original lr=2e-5 — a zh sweep at 5e-5/1e-4 was tried and both were worse). Selected by `dev` score only, not public-test comparisons.

```bash
python train.py --model_key bgem3 --context_mode none --loss weighted_bce \
  --lang_subset zh --epochs 15 --batch_size 8 --run_name F3_bgem3_zh
python train.py --model_key bgem3 --context_mode none --loss asl --lr 5e-5 \
  --lang_subset en --epochs 15 --batch_size 8 --run_name I2_bgem3_en_lr5e-5

python threshold_tune.py --run_name F3_bgem3_zh
python threshold_tune.py --run_name I2_bgem3_en_lr5e-5
python predict_and_eval.py --run_name F3_bgem3_zh
python predict_and_eval.py --run_name I2_bgem3_en_lr5e-5
python route_combine_eval.py --zh_run F3_bgem3_zh --en_run I2_bgem3_en_lr5e-5
```
Result: public test macro=0.8264/micro=0.8374; dev macro=0.8625/micro=0.8634 — **clears the challenge tier (0.82/0.83) on both metrics, on both splits.**

**Packaging Candidate C for submission** (fp16 needed — fp32 would be ~4.4GB combined, over the 4GB download budget; fp16 is ~2.2GB combined):
```bash
python package_final.py --run_name F3_bgem3_zh --out_dir ../models/zh --fp16
python package_final.py --run_name I2_bgem3_en_lr5e-5 --out_dir ../models/en --fp16
```
Then `run.sh`/`download.sh` at the repo root use `../models/zh` and `../models/en` exactly as for Candidate A (single checkpoint per route — this is why C is packageable today and B currently isn't).

## 4. Auxiliary experiments (both negative results — not part of any candidate, kept for the report)

**English domain-adaptive MLM pretraining on MultiWOZ** (unsupervised, no labels used):
```bash
python domain_adapt_mlm.py --base_model roberta-base \
  --out_dir /home/jtan/adl_hw1/domain_adapted/roberta_multiwoz_en --epochs 3
# then fine-tune from the local checkpoint via the "roberta_dapt" registry key:
python train.py --model_key roberta_dapt --context_mode none --loss asl \
  --lang_subset en --epochs 20 --lr 1e-4 --run_name H2_roberta_dapt_en
```

**Back-translation augmentation of the en route** (en→zh→en round trip via `Helsinki-NLP/opus-mt-*`):
```bash
python back_translate.py --in_path ../data/train_split.jsonl \
  --out_path ../data/train_split_en_bt.jsonl --lang en
python train.py --model_key roberta --context_mode none --loss asl --lr 5e-5 \
  --lang_subset en --epochs 20 --aug_path ../data/train_split_en_bt.jsonl \
  --run_name H4_roberta_bt_en_lr5e5
```

## 5. EDA / diagnostic utilities (no training)

```bash
python token_length_eda.py   # real tokenizer-based length distributions across candidate models
```

## Script reference

| Script | Purpose |
|---|---|
| `split_data.py` | Stratified train/dev split |
| `train.py` | Train one route (`--model_key`, `--context_mode`, `--loss`, `--lang_subset`, `--lr`, `--epochs`, `--aug_path`, ...) |
| `threshold_tune.py` | Per-class threshold tuning on a run's own dev split |
| `predict_and_eval.py` | Diagnostic evaluation of one route against `public_test_gold.csv` (never used for selection) |
| `route_combine_eval.py` | Combine two single-checkpoint routes (by `language`) and score against public test |
| `ensemble_eval.py` | Average sigmoid probabilities across multiple checkpoints for one route, re-tune thresholds, score dev + public test |
| `package_final.py` | Bundle a trained route into a self-contained local directory (`save_pretrained`) for offline `run.sh` loading; `--fp16` halves size |
| `predict_final.py` | The actual `run.sh` entrypoint — loads packaged zh/en routes, routes by `language`, writes `prediction.csv` |
| `domain_adapt_mlm.py` | Continue MLM pretraining of a base encoder on external unlabeled text |
| `back_translate.py` | Generate same-label paraphrases via round-trip machine translation |
| `token_length_eda.py` | Tokenizer-based input-length EDA across candidate models |
| `data_utils.py`, `eval_metrics.py`, `losses.py`, `models.py` | Shared library code (data loading/formatting, F1 computation, loss functions, model registry) — not run directly |
