# ADL HW1 (A1) — TODO / method bookkeeping

Running checklist of everything surveyed (Deep Research + lecture) and its status. Full configs/results go in `WORKLOG.md`; this file tracks *what's left* and indexes what's landed. Rewritten 2026-09-17 to consolidate — see git-free history in `WORKLOG.md` if older phrasing is needed.

## Project organization (established 2026-09-17)

- **Mac** (`/Users/chun-feitan/Desktop/CSIE5431/`) — Claude session work: editing docs/code, orchestrating GPU runs.
- **GPU** (`gsm-gpu2:/home/jtan/adl_hw1/`) — experiments only, disposable. Cleaned 2026-09-17: `results/` went 38GB→18GB by deleting large checkpoint/logit files for runs no longer part of any live candidate (kept only the 7 checkpoints backing candidates A/B/C — metadata for all ~30 runs is kept, just not the multi-hundred-MB weight files); removed the redundant `package/` build directory (already copied to Mac and verified).
- **GitHub** (private repo `goog-msft-fb-nflx-nvda-aapl/csie5431-adl-hw1`) — **source of truth for code and progress management going forward.** Contains this file, `WORKLOG.md`, `README.md`, `TA_QUESTIONS.md`, `FOLLOWUP_QUESTIONS.md`, `code/`, `run.sh`, `download.sh`, the spec/lecture reference docs, and the Deep Research survey. Kept **private** — the assignment bans publishing code before the deadline. Excludes provided data and packaged model weights via `.gitignore` (those don't belong in a code repo regardless of privacy).

## Current best result (this session) — two candidates, unresolved tension between them, see WORKLOG for full reasoning

**Architecture: language-routed ensemble** (route by the `language` field, given at inference time — no detection needed), not a single bilingual model. Both languages found: dropping dialogue context (utterance-only) beats every context strategy tried.

**Candidate A — simple, single checkpoint per route, selected purely on dev (most methodologically defensible)**:
- zh: MacBERT, context=none, weighted BCE, 20ep (`C2b_macbert_none_wbce_zh_ep20`)
- en: RoBERTa, context=none, ASL, 20ep, lr=2e-5 (`C3b_roberta_none_asl_en_ep20`)
- **Public test: macro=0.8057, micro=0.8238** (medium tier ✅, challenge tier ✗). **Dev: macro=0.8502, micro=0.8556** (challenge tier ✅, comfortable margin).

**Candidate B — ensembled + LR-tuned, selected partly via repeated public-test comparisons (methodologically riskier, see caveat)**:
- zh: 2-way logit-ensemble (`C2b` weighted-BCE + `C1` ASL, both lr=2e-5)
- en: 3-way logit-ensemble (`C3b`/`D1`/`D1b` — ASL at lr 2e-5/5e-5/1e-4, found via an LR sweep)
- **Public test: macro=0.8218, micro=0.8316** (challenge tier ✅ by 0.0018/0.0016 — thin margins). **Dev: macro=0.8335, micro=0.8442** (challenge tier ✅ but a smaller margin than Candidate A's dev score).
- **Caveat, stated plainly**: picking between ~10 LR/ensemble variants partly by comparing public-test scores repeatedly is a form of indirect overfitting to that fixed 500-example set, even without touching per-class thresholds on it. Dev score *dropping* (0.8502→0.8335) while public-test score *rose* (0.8057→0.8218) through this process is consistent with that risk. Not resolved — flagged for discussion.

**Key findings this session** (full evidence in `WORKLOG.md`):
1. Language-routed ensemble beats any single bilingual model by a wide margin (+0.10ish macro-F1 over bilingual XLM-R).
2. Dropping dialogue context entirely beats every context-inclusive strategy tried, on **both** languages and **both** data scales (627 en / ~1900 zh train examples) — not a low-data artifact.
3. Loss function matters much more for the smaller en route (ASL: +0.049 macro over BCE) than the larger zh route (all three losses within noise of each other there).
4. Learning rate matters a lot more for the en route (2e-5 was clearly suboptimal; 1e-4 dev flat-0.5 macro 0.79 vs. 2e-5's 0.75) than the zh route (LR sweep barely moved zh's number — more data, less LR-sensitive).
5. The public-test-vs-dev score gap is explained by the train/test language-ratio shift discovered in the original EDA: the zh route is stronger than the en route, and public test's balanced 49/51 zh/en mix (vs. train's 75/25 zh-heavy mix) exposes that gap more than dev does.
6. Free logit-ensembling of already-trained checkpoints for the same route helps on zh (both splits) but was genuinely ambiguous on en (helped dev, hurt public-test, or vice versa depending on which pair) — small-dev-sample noise (en dev n=117) makes this route's model selection unreliable to do on dev alone.

## What's left before this can be submitted

- [x] `run.sh` / `download.sh` / `code/predict_final.py` / `code/package_final.py` — **done and verified end-to-end**. `run.sh` tested on GPU with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` forced (true offline check), reproduces the exact diagnostic score (macro=0.8057, micro=0.8238) from a from-scratch packaged model load. `prediction.csv` header verified byte-identical to `sample_prediction.csv` (had to fix a `\r\n` vs `\n` line-ending mismatch first).
- [x] `README.md` — done, step-by-step train-from-scratch + run instructions.
- [ ] `report.pdf` — not started; report questions can mostly be answered directly from WORKLOG.md's findings (see mapping below).
- [x] Model+tokenizer size confirmed: zh (MacBERT) 391MB + en (RoBERTa) 481MB = 872MB uncompressed, `models.tar.gz` 765MB — well inside the 4GB download budget.
- [x] `run.sh` timing confirmed: 21s for the full 500-example public test set (both routes) — trivially inside the 2-hour budget.
- [ ] **BLOCKED ON USER**: `models.tar.gz` (765MB, at `submission_package/models.tar.gz` on the Mac) needs to be uploaded to Google Drive by the user (requires their account) and the resulting file id/share-link pasted into `download.sh`'s `GDRIVE_FILE_ID` placeholder. Nothing else in the pipeline depends on this being done immediately — can continue other work in the meantime.
- [ ] Revisit `TA_QUESTIONS.md` if/when TA responds — still unanswered as of this session. We proceeded conservatively (dev-only threshold tuning, no external-dataset pretraining, no train-derived augmentation); item 1 (can we tune on public-test labels?) is the one most likely to unlock more performance if answered "yes."

## Report-question mapping (draft pointers into WORKLOG.md)

- **Model variants**: Phase A (mBERT/XLM-R/MacBERT/RoBERTa/SpanBERT comparison) + the routing-vs-bilingual finding.
- **Does context help**: Phase B — no, consistently, across both languages and data scales; see the "context doesn't help" sections for B1/B3/B5 (en) and B6 (zh).
- **Long inputs**: Phase B's truncation comparison (headtail vs naive lastk-4) — headtail recovers +0.144 macro-F1 over naive truncation even though neither beats dropping context.
- **Chinese vs English**: Phase A's asymmetric specialization-gain finding (MacBERT +0.098 over bilingual on zh vs RoBERTa +0.019 over bilingual on en) + the language-routing architecture itself + the dev-vs-public-test gap mechanism.
- **Additional exploration candidates** (not yet written up as a dedicated section, but data exists): (a) the language-routed-ensemble discovery itself is probably the headline finding for this section; (b) flat-0.5 vs. tuned-threshold macro/micro trade-off, data already in every run's `thresholds.json`; (c) per-class ΔF1 from context — data exists in B-phase per-class breakdowns, not yet assembled into a dedicated figure.

## Models — surveyed, availability, status

| Model | HF ID | Source | Role | Status |
|---|---|---|---|---|
| mBERT | `bert-base-multilingual-cased` | lecture | bilingual baseline | done (A1) — lost to XLM-R |
| XLM-R base | `xlm-roberta-base` | extension of lecture's XLM | bilingual baseline | done (A2/A2b) — best single bilingual model, but beaten by routing |
| Chinese MacBERT base | `hfl/chinese-macbert-base` | extension | **zh route (selected)** | done, finalized as zh route |
| RoBERTa base | `roberta-base` | lecture | **en route (selected)** | done, finalized as en route |
| SpanBERT base | `SpanBERT/spanbert-base-cased` | lecture | en ablation | done (A5) — confirmed poor fit, excluded |
| DeBERTa-v3 base | `microsoft/deberta-v3-base` | extension | en route candidate | done (A4b) — lost to RoBERTa under identical recipe, not chased further |
| Original XLM (Lample & Conneau) | `FacebookAI/xlm-mlm-17-1280` or similar | lecture | literal lecture-XLM data point | not started, feasibility unchecked (lang-id input handling may not plug into `AutoModelForSequenceClassification` cleanly) |
| mDeBERTa-v3 base | `microsoft/mdeberta-v3-base` | DR extension | bilingual alternative | not started — moot now that routing beats bilingual |
| BGE-M3 | `BAAI/bge-m3` | DR extension | long-context ablation | not started, low priority — context has consistently hurt, so a long-context model is unlikely to help |
| Qwen2.5-0.5B/1.5B-Instruct | HF | DR extension | Additional Exploration candidate | not started, low priority |

## Datasets — surveyed, none usable as direct label transfer

| Dataset | License | Considered use | Status |
|---|---|---|---|
| MultiWOZ 2.1/2.2 | Apache-2.0 | Domain-adaptive MLM pretraining, en route | **tried, negative result** (2026-09-17, run on user's explicit "assume allowed" instruction — see WORKLOG): dev tuned macro 0.8160 vs. vanilla 0.8196 at the same LR, and clearly behind the actual en champions. Not adopted. |
| Schema-Guided Dialogue (SGD) | CC BY-SA 4.0 | Auxiliary dialogue-act pretraining | not tried — MultiWOZ's negative result plus the shared "not a label-signal problem" explanation makes this lower priority now |
| MASSIVE | CC BY 4.0 | Multilingual transfer / back-translation seed | not tried, same reasoning |
| JDDC / ECD | gated / unclear terms | Chinese e-commerce domain MLM adaptation | deprioritized — licensing friction not worth the timeline risk |
| SalesLLM benchmark | — | **BANNED by name in PA1.md** | excluded |
| MultiSense/SaleIntent_bert | — | **BANNED — trained on another sales-intent dataset** | excluded |

## Techniques — status

| Technique | Status |
|---|---|
| Per-class threshold tuning (macro-F1 s.t. micro-F1 floor) | **done, both routes, biggest single lever every time it was tried** |
| Weighted BCE | done both routes — won on zh (marginally), lost to ASL on en |
| Asymmetric Loss | done both routes — won on en (clearly), tied with weighted BCE on zh |
| Context dropping (vs. lastk/headtail/naive) | done both routes — "none" won consistently |
| Head+tail token-level truncation | done (en route) — validated as a technique (+0.144 over naive truncation) even though it didn't end up in the final config |
| Language routing | **done — the single biggest architectural win this session** |
| Epoch scaling (10→20) | done both routes — real gains both times, diminishing by epoch ~15-20 |
| MLSMOTE / oversampling | **deliberately skipped** — DR consensus says poor fit for multi-label at n=3000 |
| Back-translation / paraphrase augmentation | **tried (en route), negative result** — 2-for-2 neutral-to-negative vs. no-augmentation at two learning rates (dev tuned macro 0.8364→0.8028 at lr5e-5, 0.8196→0.8152 at lr1e-4). Not adopted. |
| Domain-adaptive MLM pretraining on external corpora | **tried (MultiWOZ, en route), negative result** — see Datasets table above. |
| BGE-M3 fp16 packaging | **done** — halves 2.2GB/route to 1.1GB/route (2.2GB total both routes), fits the 4GB budget. Not yet verified fp16 doesn't cost accuracy — should spot-check before treating as final. |
| Language-balanced sampling (`--balance_lang`) | implemented, **not needed** — routing made this moot (each route only ever sees its own language) |

## Round-2 exploration queue (2026-09-17, per explicit instruction: work through everything unblocked, document each, then discuss)

Everything below is unblocked by the TA (doesn't need external-dataset or augmentation approval). Working through in this order:

- [x] zh-route context sweep with lastk-4/headtail (B7/B8), same recipe as the final zh route (weighted BCE, 20ep) — **confirms the en-route pattern exactly**: none(0.8692) > headtail(0.8586) >> lastk-4/naive(0.6522). Long-inputs report figure now symmetric across both languages.
- [x] rerun mBERT (A1) with 20 epochs — **corrected an earlier wrong conclusion.** mBERT-20ep: dev tuned macro=0.7821, public-test tuned macro=0.7547 — **beats XLM-R-10ep (0.7587/0.6847) outright.** The original "XLM-R wins Phase A" call was comparing an undertrained mBERT (6ep, wrongly assumed plateaued) against a properly-trained XLM-R. Doesn't change the final architecture (routing still beats any single bilingual model), but is a real correction to the "model variants" report answer — logged plainly in WORKLOG, not quietly fixed.
- [x] `microsoft/mdeberta-v3-base` — tried both routes, **loses clearly** to current champions (zh: 0.8255 vs MacBERT's 0.8692; en: 0.8067 vs RoBERTa's 0.8364). Not pursued further.
- [x] `BAAI/bge-m3` — **new best model, first try, both routes.** zh dev tuned=0.8667 (~tied w/ MacBERT), zh public tuned=**0.8593** (beats MacBERT's 0.8513). en dev tuned=**0.8392** (new best, beats RoBERTa's 0.8364), en public tuned=**0.7950** (beats RoBERTa's best single 0.7770). **Routed (single model each route, no ensembling): public macro=0.8260/micro=0.8273, dev macro=0.8605/micro=0.8618.** Clears challenge tier on macro (public) and both metrics (dev) — and unlike the earlier ensembled candidate, dev and public test *agree* this is better (both went up together), which is a much cleaner signal than the ensembling episode where they diverged. **New leading candidate for final submission — see WORKLOG for full writeup and the methodological comparison against the ensembled candidate.**
- [x] Original XLM (`FacebookAI/xlm-mlm-17-1280`) — **checked, not pursuing.** License is CC-BY-NC-4.0 (non-commercial — fine for coursework but worth noting), and the model card explicitly requires language-embedding/`lang_id` inputs for correct use, doesn't have a ready sequence-classification config, and isn't a clean drop-in to `AutoModelForSequenceClassification` the way every other model tried this session was. Given the routing architecture with per-language encoders already beats every bilingual model tried by a wide margin, and this would need non-trivial custom head/input-format work for a model whose successor (XLM-R) we've already tried and out-ensembled, deprioritizing rather than building custom plumbing for it.
- [x] Qwen2.5-0.5B-Instruct, properly fine-tuned (not zero-shot) as a classifier on both routes — zh dev tuned=0.8355 (beats mDeBERTa, loses to MacBERT/BGE-M3), en dev tuned=0.7933 (comparable to RoBERTa's un-tuned baseline, loses to RoBERTa's LR-tuned best and BGE-M3). Overfits hard and fast on both routes (train loss → ~0 by epoch 8-14). Competitive but not competitive enough to win either route. Moderate, not dramatic, support for "encoders beat small decoders" — weaker than the DR consensus implied.
- [x] Verify XLM-R / MacBERT exact license text against the live HF model cards — **done.** `xlm-roberta-base`: **MIT** (confirms Kimi/Perplexity's Deep Research claim, refutes Gemini's guessed "CC BY-NC 4.0" — glad we checked ourselves rather than trusting either). `hfl/chinese-macbert-base`: **Apache-2.0** (confirmed as stated).
- [x] Dedicated per-class ΔF1-from-context figure/table — assembled in WORKLOG.md. Reveals a language-specific exception hidden by the aggregate finding: on English only, `Compare_Competitor`/`Too_Expensive` are the two classes where context still helps (matching the original EDA hypothesis for exactly those two); on Chinese even those two improve without context.

Blocked, not attempting until TA answers: JDDC/ECD (deprioritized on licensing grounds independent of the TA question), SGD/MASSIVE auxiliary pretraining (deprioritized given MultiWOZ's negative result and shared explanation — see below).

## Round 3 (2026-09-17) — proceeded on external-dataset pretraining and augmentation, user said assume allowed

Per explicit instruction: assumed permission for external-dataset auxiliary pretraining and train-derived augmentation (still unanswered by the TA — revisit if they say otherwise), and confirmed the 4GB budget is about the grading machine's `download.sh`, not GPU disk space (GPU never a constraint).

- [x] **BGE-M3 fp16 packaging** — confirmed 1.1GB/route (2.2GB total), fits the 4GB budget. `package_final.py --fp16` implemented.
- [x] **English domain-adaptive MLM pretraining on MultiWOZ** (`code/domain_adapt_mlm.py`, new) — 128K unlabeled English utterances, 3-epoch MLM continued pretraining of `roberta-base`, then fine-tuned as usual. **Negative result**: dev tuned macro 0.8160 vs. vanilla 0.8196 at the same LR — no better, arguably slightly worse.
- [x] **Back-translation augmentation of the en route** (`code/back_translate.py`, new, uses `Helsinki-NLP/opus-mt-en-zh`/`opus-mt-zh-en`) — doubled the en training set (627→1254). **Negative result, 2-for-2**: worse than no-augmentation at both learning rates tried (lr5e-5: 0.8364→0.8028; lr1e-4: 0.8196→0.8152).
- **Conclusion**: both previously-blocked techniques turned out neutral-to-negative once tested, not missed wins. Working theory: the en route's real bottleneck is labeled-signal scarcity (rare classes, only 627 examples), which neither unlabeled-text pretraining nor label-preserving paraphrase augmentation actually addresses. **Standing champions unchanged**: BGE-M3 (best overall), vanilla RoBERTa lr=5e-5 no-augmentation (best RoBERTa-only en config).

## Round 4 (2026-09-23) — BGE-M3 LR sweep + round-2 Deep Research backlog

- [x] **BGE-M3 LR sweep** — zh: 2e-5 (original) remains best, 5e-5/1e-4 both worse. **en: lr=5e-5 is a new best** (dev tuned macro 0.8445 vs. 1e-4's 0.8392). Candidate C updated: public test macro=0.8264/micro=0.8374, dev macro=0.8625/micro=0.8634 — **now clears challenge tier (0.82/0.83) on both metrics, on both splits**, selected on dev only (no public-test-driven selection, unlike Candidate B).
- [x] Round-2 Deep Research prompt sent (Gemini extended thinking + a Perplexity/Compass-style source), both read and synthesized. Backlog below, ordered by the sources' own priority — **working one at a time, not in parallel**, per explicit instruction.

### Backlog (in priority order — do these one-by-one, log each)

1. [x] **Multi-seed reliability check on the en route** — confirmed the literature's warning: 5 seeds of the same config span tuned dev macro 0.8387–0.8534 (std≈0.005), the same order of magnitude as the earlier LR-sweep "win." **Fix applied, not just diagnosed**: 5-seed ensemble (free, checkpoints already trained) → dev tuned macro **0.8714** (beats every individual seed), public-test-en tuned macro **0.8134** (+0.024 over the best single seed). **Candidate C updated: public test macro=0.8372/micro=0.8466, dev macro=0.8690/micro=0.8696 — clears challenge tier with real margin now (was razor-thin before).** Biggest single gain this session. See WORKLOG for full numbers.
2. [x] **LP-FT (linear-probe-then-fine-tune)** — `train.py --lp_epochs N` implemented (freeze backbone N epochs, then unfreeze). Single run (5 probe + 10 FT epochs) beats the single-seed baseline: dev tuned macro 0.8523 vs. 0.8445 (+0.0078), public-en tuned 0.7942 vs. 0.7896 (+0.0046) — real but modest, roughly at the edge of the measured seed-noise band. **Tried as a 6th ensemble member (diversity beyond same-recipe seeds) — didn't help**, 6-way ensemble slightly below the 5-way (dev 0.8703 vs 0.8714, public 0.8087 vs 0.8134). **Not adopted** — 5-seed vanilla ensemble remains the standing best. Flag kept in codebase for future revisiting (e.g. combined with item 3's stability recipe).
3. **Stability recipe**: confirm AdamW bias correction isn't disabled (should be fine, it's the HF default), add longer warmup (10-20% vs. our current 6%), and try re-initializing the top 1-6 transformer layers before fine-tuning the en route. Explains our own LR-sensitivity finding rather than just working around it.
4. **Distribution-Balanced Loss / CB-NTR** (Huang et al. 2021, EMNLP, `github.com/Roche/BalancedLossNLP`) as a replacement for ASL on the en route — purpose-built for long-tailed multi-label text, direct upgrade path for `Too_Expensive`/`Compare_Competitor`. Also cheap to test alongside: post-hoc logit adjustment (Menon et al. 2021).
5. **zh→en sequential fine-tuning or joint multilingual training** (no machine translation) — fine-tune on the abundant zh data first (or jointly), then continue/specialize on en. Tests the "borrow from Chinese" hypothesis without translation-artifact risk, before reaching for MT.
6. **Translate-train** (if #5 underperforms) — translate the ~1900 zh training examples into English (NLLB-200-distilled-600M recommended over the MarianMT models we already have, for quality), inject as *new* en training content (not paraphrases — this is fundamentally different from the back-translation we already tried and that failed twice). Tag or down-weight translated examples; keep genuine en at ≥1:1 ratio; spot-check the two rare classes' translations specifically.
7. **New model candidates**: `BAAI/bge-large-en-v1.5` (en route) / `BAAI/bge-large-zh-v1.5` (zh route) — smaller, language-specialized, retrieval-only (clean compliance), matches our per-language-route design philosophy. `intfloat/multilingual-e5-large` as a same-architecture (XLM-R-large) comparison to BGE-M3 (needs `query:`/`passage:` input prefixes — check this is handled correctly).
8. **Explicitly ruled out** (compliance risk per round-2 research, not attempting): `jinaai/jina-embeddings-v3` (ships a dedicated classification LoRA adapter — directly conflicts with our "not trained on other intent/classification data" reading of the rules), `Qwen/Qwen3-Embedding-0.6B` (trained on synthetic classification data **and** requires transformers ≥4.51, which we don't have — hard blocked either way), `intfloat/multilingual-e5-large-instruct` (fine-tuned on classification-style data, medium risk).
9. Lower priority / only if the above stall: SetFit-style contrastive fine-tuning reimplemented in plain PyTorch (no `sentence-transformers`/`setfit` packages — not on our allowed list) on BGE-M3's native CLS+L2-norm embeddings; sigmoidF1 loss (tuning-sensitive per the source, secondary bet); PET/cloze-prompt framing for the two rare classes specifically.

All public/ungated — no HuggingFace or GitHub tokens needed for anything on this list currently. Will flag immediately if that changes.
