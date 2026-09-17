Here's the full survey, structured by your six questions. I read your brief carefully — the interesting constraints (10GB VRAM, 4GB download, offline grading, the zh/en shift) drive most of what follows.

---

## 1. EDA — what you're missing

Your EDA is already solid (frequency, cardinality, co-occurrence, language split, context stats). What a strong practitioner would add:

**Token-length distributions, per language, per candidate tokenizer.** Character counts are misleading for model selection because Chinese tokenizes ~1–1.5 chars/token with BERT WordPiece/SentencePiece while English is ~4 chars/token. Concrete things to compute:
- For `bert-base-multilingual-cased`, `xlm-roberta-base`, and `bert-base-chinese`: token counts of (utterance only) and (context + utterance concatenated), reported separately for zh and en, with percentiles (p50/p90/p99/max). This directly tells you what fraction of examples actually blow a 512 window — with your p99 of ~3,956 chars (mostly zh, presumably), a Chinese-specific tokenizer may keep p99 under 512 while mBERT doesn't.
- Length-conditioned label analysis: are long-dialogue examples disproportionately `Confirm_Order` / `Compare_Competitor` (negotiation-closing intents that plausibly need accumulated context) vs. short-utterance `Ask_*` questions? This determines whether the long tail matters for the classes that matter.
- Per-class, per-language label rates with confidence intervals on the 500-example public test (n=500 means ±3–4 points per class is noise; your "within ~3.5 points" observation may be entirely consistent with no per-class shift at all).

**Near-confusable pair forensics.** Before modeling, hand-annotate ~50 examples each for `Doubt` vs `Need_More_Evidence` vs `Ask_Spec`, and `State_Need` vs `Ask_Spec`. Extract character/word n-grams per label and look at mutual-information-ranked discriminators *within* the confusable triple, not globally — global ranking will just surface "多少钱/price" for `Ask_Price`. Also check whether the same utterance text appears with different labels across rows (annotation noise / soft boundaries), which caps achievable F1.

**Co-occurrence as structure.** Your top pairs are all among the four frequent classes — that looks like a "generic engagement" cluster rather than semantic coupling. Check whether conditioning helps: P(`Need_More_Evidence` | `Doubt`) vs base rate. If the co-occurrence graph is nearly rank-one (one big frequent-component), label-correlation modeling (classifier chains, GCN over labels) will add little; if there are real conditional dependencies (e.g., `Confirm_Order` strongly implies recent `Too_Expensive` resolution), a cheap post-hoc calibration of P(y_j | top-k other labels) could be worth +Macro-F1.

**Distribution-shift red flags beyond language ratio.** Since the assignment explicitly warns of shift, check on public test vs train: (a) domain mix (e-commerce vs financial/insurance wording — insurance products have long model names and policy jargon); (b) context-turn count distribution (test median 4 like train?); (c) empty-context rate (10.6% vs 9.9% — fine); (d) utterance-only length; (e) duplicate/near-duplicate utterances between train and public test (dedup strategy); (f) label cardinality distribution. The language-ratio shift you found is real but benign in expectation (English is high-resource; zh→en transfer is the "easy" direction — see §6). The nastier possibility is a *domain* shift hidden inside it.

**One cheap diagnostic worth doing now:** fine-tune a quick baseline on utterance-only vs context+utterance and look at *which classes* gain from context. That answers the report's "does dialogue context help" question empirically and tells you where to spend long-input effort.

---

## 2. SOTA/practical model choices

**Constraint math first.** Fine-tuning memory ≈ weights + Adam states + activations. On 10GB you can comfortably fine-tune any base-sized encoder (~100–280M params) at 512 tokens with fp16 + reasonable batch; a 560M encoder is feasible but tight (needs gradient checkpointing, small batches, grad accumulation). Inference over 500 test examples is trivial for anything below ~3B params. The 4GB download budget comfortably fits a large encoder fp16 (~1.2GB) plus a backup model, or one 1.5B decoder fp16 (~3.1GB) with nothing else.

**Multilingual encoders (primary candidates):**

| Model | Params | Disk (fp32/fp16) | License | Notes |
|---|---|---|---|---|
| `bert-base-multilingual-cased` | ~179M | ~0.7GB / ~0.36GB | Apache-2.0 | Weakest zh+en option; WordPiece splits Chinese aggressively |
| `xlm-roberta-base` | ~278M | ~1.1GB / ~0.56GB | CC BY-NC 4.0* | Strong default; SentencePiece handles zh well; 100+ languages |
| `xlm-roberta-large` | ~561M | ~2.2GB / ~1.1GB | CC BY-NC 4.0* | Best multilingual ceiling; ~2× fine-tune compute |
| `intfloat/multilingual-e5-base` | ~278M | ~1.1GB | MIT | XLM-R architecture, retriever-tuned; fine for classification heads too |
| `BAAI/bge-m3` | 569M | 2.27GB fp32 (official table) | MIT | 8192-token window (!), 100+ languages, SOTA multilingual retrieval in its class   |

\* Double-check the XLM-R license on the HF card — I believe it's CC BY-NC 4.0 (fine for coursework; would matter commercially). BGE-M3 loads with plain `transformers` (XLM-RoBERTa architecture), needs no `FlagEmbedding`, and its MIT license + 2.27GB fp32 size + native 8192 context makes it arguably the best single-model fit for your constraints — the one caveat is it's trained for retrieval, so you should MLM-adapt it briefly on your dialogue text before the classification head (cheap, works well in practice).

Evidence multilingual encoders do this job: XLM-R fine-tunes are the standard strong baseline for multilingual long-text classification; in one recent controlled study, fine-tuned XLM-R-base hit weighted macro-F1 0.76 vs 0.79 for a Longformer variant, and **truncation at 512 vs 1024 vs 2048 tokens made no difference** (0.76 all)  — encouraging for your 512-token plan (details in §4).

**Chinese-specialized (for zh ablations):**
- `bert-base-chinese` (~102M, Apache-2.0) — vocabulary and tokenizer far more token-efficient on zh than mBERT's.
- `hfl/chinese-roberta-wwm-ext` (~110M, Apache-2.0 per card — verify) — whole-word-masking, consistent small gains on Chinese classification benchmarks.
- `hfl/chinese-macbert-base` (~102M) / `-large` (~325M) — MacBERT's MLM-as-correction pretraining tends to help short-text classification; the standard Chinese classification baseline.
- `nghuyong/ernie-3.0-base-zh` (~113M, Apache-2.0 — verify) — KG-enhanced; strong on Chinese NLU but heavier tokenizer.

**English-specialized (for en ablations):** `bert-base-uncased` (110M), `roberta-base` (125M, MIT), `microsoft/deberta-v3-base` (184M, MIT) — DeBERTa-v3 is the strongest per-parameter English encoder; note it needs `sentencepiece` (a sub-dependency of transformers, allowed).

For the report's "Chinese vs English" question: the cleanest design is (a) bilingual XLM-R/BGE-M3 as the single-model answer, (b) zh-only and en-only models trained on their respective subsets as ablations, (c) per-language evaluation of the bilingual model. Expect the bilingual model to beat monolingual ones on each language individually — that's the interesting, reportable finding.

**Small decoder LLMs as classifiers.** The literature says: fine-tuned encoders still win for this. Across sentiment/stance/emotion tasks, "smaller, fine-tuned LLMs (still) consistently and significantly outperform zero-shot prompted GPT-3.5/GPT-4/Claude, especially for specialized, non-standard tasks" . An independent 32-experiment comparison of exactly the models you'd consider (Qwen2.5-0.5B/1.5B-Instruct, Gemma-2-2B vs BERT-base/DeBERTa-v3) found DeBERTa-v3 won 3 of 4 tasks when training data exists, and encoders are ~20× faster at inference (277 vs ~12 samples/s) . Zero-shot small LLMs only edge out encoders on adversarial tasks (ANLI) — not your setting. Verdict: Qwen2.5-1.5B is a credible *auxiliary* (e.g., offline paraphrase augmentation, or an ensemble member), not the primary classifier. Also: multi-label + bilingual + 10 intent definitions is exactly the kind of nuanced, non-standard label semantics where zero-shot prompting degrades most .

---

## 3. Public datasets that could legally help

Honest read: **the label schemas are all far enough from your 10 sales-intent labels that none is useful as labeled training transfer under your rules.** The value is in (a) domain-adaptive MLM pretraining on dialogue text, and (b) augmentation corpora.

| Dataset | What | Size | License | Usefulness here |
|---|---|---|---|---|
| **MultiWOZ 2.1/2.2** | Task-oriented dialogues w/ user+system dialogue acts; ~60% of system turns carry multiple acts — genuinely multi-label | 10,438 dialogues, 115k turns  | MIT (check release) | DA tags (inform/request/confirm…) don't map to sales intents, but English dialogue text is good MLM adaptation data; multi-act annotation format is a nice report comparison |
| **SGD (Schema-Guided Dialogue)** | 20k+ dialogues, 16–20 domains, active-intent per user turn | train 16,142 / dev 2,482 / test 4,201  | **CC BY-SA 4.0**  | Best legal fit: intent-prediction benchmark, varied domains. Schema intent labels still don't match yours — use as auxiliary intent-transfer *cautiously* (it's not a "sales-intent dataset," but clear it with the TA to be safe) |
| **JDDC** | Real Chinese e-commerce customer-service dialogues | >1M dialogues, 20M utterances, 289 intents  | Requires application to JD (email form) — likely too slow for your timeline | Domain text (zh e-commerce!) is ideal for MLM adaptation if you can get it; its 289 after-sales intents are noisy in-house labels (93% precision) — do not use as classifier training signal |
| **Multi3WOZ** | MultiWOZ in 10 languages incl. Chinese, parallel, same DA annotations | multi-parallel | CC-style (check) | Useful for a cross-lingual DA ablation; niche |
| **MASSIVE** | Intent classification, 51 languages, AmazonAlexa domain | ~1M utterances | CC BY-SA 4.0 | Multi-label? No (single-label, 60 intents). Good multilingual-intent benchmark for the report; its zh+en parallel structure could seed back-translation augmentation |

**Augmentation pipelines within your package list.** You can't call APIs at inference, but during *development* you can generate augmented training data however you like as long as final inference is offline and the rules are respected. Practical options: (1) Qwen2.5-1.5B-Instruct (offline, downloaded once) to paraphrase/translate your own train utterances — a zh↔en translation-consistency augmentation also directly attacks the language-ratio shift; (2) back-translation needs a translation model — small Helsinki-NLP opus-mt zh↔en models (~300MB) are on HF and loadable with plain transformers; (3) code-switching word substitution using multilingual static embeddings — the literature shows this improves cross-lingual robustness . Be careful: augmented data must not leak test content — derive everything from train.jsonl only.

---

## 4. Long-input handling

Your tail is real (p99 ~3,956 chars) but bounded. Strategy ranking for your case:

1. **Utterance-only and last-K-turns baselines first.** Median context is 4 turns; encode `[last 1–2 user turns + current utterance]` as a compact strong baseline. This sidesteps the window problem entirely and gives you the clean "does context help?" ablation: utterance-only → utterance+last-K → full-window.
2. **Truncation policy when concatenating full context: head+tail, never head-only.** Keep the current utterance + most recent turns verbatim (recency matters — intents like `Confirm_Order` and `Too_Expensive` resolve *recent* offers), and, if budget remains, a leading "[context summary]" of earlier turns. Head+tail truncation is standard practice and empirically robust. Practical recipe: reserve ~60% of the 512 budget for the tail (utterance + last turns), fill the rest from the head.
3. **Empirical reassurance on truncation:** in controlled long-document classification experiments, extending max length from 512→1024→2048 changed macro-F1 not at all (0.76 across all)  — i.e., for classification, *where you cut* usually matters more than *whether you keep everything*, and 512-token classification with sensible truncation is a legitimate SOTA-competitive setup. Fine-tuned XLM-R-base (0.76) was only modestly behind a 4096-window Longformer (0.79) on genuinely long documents .
4. **Long-window checkpoints.** `markussagen/xlm-roberta-longformer-base-4096` exists and is loadable via transformers (Longformer is in your allowed stack) ; BGE-M3 natively supports 8192 . Given (3), I'd treat these as an ablation, not the primary path — 4096-token attention is ~8× the compute of 512 for marginal gains on a task where the signal is usually local.
5. **Hierarchical/turn-level encoding** (encode each turn, mean/attention-pool, then a small classifier over turn embeddings): clean, window-proof, and a nice report section, but two-stage training and modest gains. Only worth it if your "does context help" experiment shows the long tail is where errors concentrate.

Which intents plausibly need long context: `Confirm_Order` (needs the accumulating negotiation), `Compare_Competitor` and `Too_Expensive` (price objections often reference earlier quotes), and disambiguating `Doubt` (what is being doubted?). `Ask_Price`/`Ask_Spec`/`Ask_Service` are usually decidable from the utterance alone. Verify this against your data in the §1 diagnostic — it makes a great report figure (per-class ΔF1 from adding context vs. dialogue length).

---

## 5. Imbalanced multi-label training & Macro-F1

**Losses.** Standard BCE-with-logits is the baseline. Given rare classes at 6.6%:
- **ASL (asymmetric loss)** — decouples positive/negative focusing (γ⁺, γ⁻), down-weights easy negatives, and is the established multi-label long-tail loss: it beat focal loss, LDAM and CE on long-tailed multi-label benchmarks  and improved macro-F1 over focal in single-label long-tail settings (77.6 vs 76.1) . It's ~15 lines of PyTorch, no extra deps — easy win and a good ablation. (Originates in the image literature; a 2024 text-specific "weighted ASL" exists but the vanilla version transfers fine.)
- Focal loss is a reasonable second choice; class-weighted BCE is the floor.
- **Skip heavy resampling.** The literature explicitly notes re-sampling is "hardly applicable" to multi-label data because a sample containing both head and tail labels can't be cleanly over/under-sampled . If you want a sampling lever, use mild label-aware batch composition (ensure each batch contains some positives for the rarest 3–4 classes) rather than MLSMOTE.

**Threshold tuning — your biggest single lever, and your biggest risk.** Theory you must know: for a calibrated classifier, the F1-optimal threshold is **half the achievable F1**, not 0.5; and crucially, the thresholds that maximize macro-F1 vs micro-F1 for the *same* probabilities can be wildly different — for rare labels with weak classifiers, macro-F1-optimal thresholding predicts *everything* positive (the "Platypus problem")  . Practical consequences:
- Do **per-class threshold search on a held-out split** (grid 0.05–0.7), optimizing a *joint* objective like `min(macro-F1, micro-F1)` or maximizing macro-F1 subject to micro-F1 ≥ threshold — since both are graded and micro-F1 is less sensitive to rare labels, thresholds tuned for macro alone can sink micro. With 3 rare classes at ~6.6% and n=500 public test, per-class F1 estimates are noisy: tune on a stratified dev carve-out (e.g., 500 of your 3000 train rows, stratified by language × cardinality), and consider shrinking thresholds toward a global value (ridge-regularized per-class thresholds, or tune only the 4 rarest classes and share one threshold for the frequent ones).
- Report both the flat-0.5 and tuned-threshold numbers in the report — the gap is itself an "Additional Exploration" finding. (An illustrative case: per-class tuning moved macro-F1 from 0.447→0.68 in a controlled example .)
- Threshold-tuning on the *public test* is legal (gold labels are released) but dangerous — the private test may shift; prefer dev-tuned thresholds, validated on public test.

**Macro/micro pitfall summary:** micro-F1 rewards precision on frequent classes; macro-F1 is dominated by your 3 rare classes (`Compare_Competitor`, `Too_Expensive`, `State_Need`). Concretely: if `State_Need` F1 is 0, macro-F1 is capped at 0.90. So the rare classes deserve: ASL or class weights, possibly rare-class-focused threshold ranges, and error analysis aimed specifically at them.

---

## 6. Robustness to the train/test language-ratio shift

Your read (proportion shift, not semantics shift) is the most plausible one, and the good news is the direction is favorable: transferring from a higher-resource/higher-data setting toward a well-resourced language (English) is the easy direction — controlled cross-lingual fine-tuning studies find *downward* high→low transfer gives ~+15% while upward transfer degrades, and English is the strongest-represented language in every multilingual encoder . So a bilingual encoder fine-tuned zh-heavy should still do well on en at test time. Still, do these:

1. **Use a genuinely multilingual encoder with balanced zh+en pretraining** (XLM-R/BGE-M3), not mBERT (zh is underrepresented in mBERT's Wikipedia-heavy corpus) and certainly not a monolingual model for the main system.
2. **Language-balanced batch sampling / slight en up-weighting during fine-tuning.** Simple, effective, and directly targets the shift. A re-weighting factor of ~2× on en examples (to match test proportions) is a defensible starting point; tune on dev-per-language F1.
3. **Translation-pair augmentation:** translate a slice of zh train to en (or paraphrase en→zh) so the model sees near-duplicates across languages — this is the practical version of code-switch augmentation, which is shown to reduce cross-lingual representation gaps . An even cheaper trick: with probability p, swap an utterance with its zh↔en counterpart (same label) during training — forces language-invariant intent features.
4. **Evaluate per language** throughout. If the bilingual model is much weaker on en at dev, you still have the per-language fallback (zh-only + en-only models, route by detected language — detection is just a Unicode range check for zh vs en).
5. **Flag other readings of the "different distribution" hint:** (a) *domain* mix shift (insurance-style dialogues heavier at test) — check product/policy vocabulary in public test; (b) *label* shift — you already checked public test label rates, do it per language (maybe en test examples have more `Confirm_Order` because English dialogues are later in the sales funnel); (c) *context* shift — test dialogues may be longer; (d) *style* shift — translated-vs-native text differences (if en test looks machine-translated, that's detectable via n-gram regularity). Each is checkable on your public test in an afternoon.

---

## Prioritized recommendation (what I'd actually do first)

1. **EDA add-ons (§1):** token-length distributions for XLM-R vs chinese tokenizers; per-class context-gain diagnostic with a quick utterance-only XLM-R-base baseline; shift checks (per-language label rates, domain vocabulary, length).
2. **Primary model:** `xlm-roberta-base` fine-tuned on `[context head+tail truncated to 512] + utterance`, BCE/ASL, fp16 — this alone should clear basic/medium. Download `bge-m3` (2.27GB, MIT, 8192 window) as the second model for the challenge tier and long-context ablation; brief MLM adaptation on your dialogue text first.
3. **Thresholds:** per-class (or per-class-group) threshold tuning on a stratified dev split with a joint macro/micro objective — likely the largest cheap Macro-F1 gain.
4. **Language-shift handling:** en up-weighting (~2×) + zh↔en pair augmentation if time allows; per-language eval always reported.
5. **Loss/ablation ladder for the report:** BCE → class-weighted BCE → ASL; utterance-only → last-K → full-512; flat 0.5 → tuned thresholds. Each cell is cheap (3000 examples trains in minutes per epoch on a 3080 Ti).
6. **Small LLM only as auxiliary:** Qwen2.5-1.5B offline paraphrase augmentation, or a 3-model ensemble (XLM-R + MacBERT + BGE-M3) via logit averaging if thresholds alone stall below the challenge bar.
7. **Additional Exploration suggestion:** the "Platypus" threshold dynamics — show that macro-F1-optimal thresholds over-predict rare labels and quantify the macro/micro trade-off frontier on your data. It's novel-ish, cheap, and directly relevant to the grading metric. Runner-up: the per-class ΔF1-from-context vs dialogue-length analysis.

**Materially unsure about — verify yourself:**
- Exact HF license fields for `xlm-roberta-*` (I believe CC BY-NC 4.0), `hfl/chinese-roberta-wwm-ext`, `hfl/chinese-macbert-*`, `nghuyong/ernie-3.0-*`, and MultiWOZ releases — read the cards before `download.sh` is finalized.
- Whether auxiliary training on SGD/MASSIVE intent labels is within the "clear it first" spirit of the rules — ask the TA; MLM-only adaptation is unambiguously safe.
- Whether `run.sh`'s 2-hour budget includes only inference or also your threshold-tuning (tuning on public test labels may be considered "touching test labels" if the grader is strict — dev-split tuning is the safe default).
- Whether translated/paraphrased training data derived from train.jsonl counts as "annotating more data from the original source" — get a ruling before investing in the augmentation pipeline.
- True token-length blowout rate for XLM-R on your zh context+utterance — my guess is 10–20% exceed 512, but measure it.