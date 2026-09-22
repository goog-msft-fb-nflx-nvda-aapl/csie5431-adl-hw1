# Research brief, round 2: pushing past a strong baseline — multi-label sales-intent classification

This is a follow-up to an earlier survey (results in `deep_research_response/`, four sources: Gemini, Kimi, Perplexity, Qwen). That round covered the basics well (model shortlist, loss functions, threshold tuning, truncation strategies) and those recommendations mostly worked. This round is narrower and should go deeper: real academic literature (arXiv, ACL/EMNLP/NAACL anthology, not just blog-level summaries), production GitHub repos, and HuggingFace model/dataset search — specifically aimed at the concrete bottleneck described below. Please cite sources (papers, repo links, HF model cards) with links.

## Complete problem statement

Grad NLP course assignment: **multi-label intent classification** (10 possible labels per example, 0 to K apply) on bilingual (Chinese/English) sales-conversation utterances, with optional prior dialogue context. Metrics: Macro-F1 and Micro-F1, both graded against fixed thresholds simultaneously — Basic (0.62/0.73), Medium (0.76/0.79), Challenge (0.82/0.83).

**Hard constraints**:
- Inference hardware: Ubuntu 20.04, 32GB RAM, RTX 3080 Ti (10GB VRAM), 20GB disk, **no network access** after a one-time `download.sh` step.
- `download.sh` may fetch at most **4GB total**, must finish in 1 hour at ~10MB/s.
- Allowed packages only: Python 3.10/3.12, PyTorch 2.11.0, scikit-learn, tqdm, numpy, pandas, transformers 4.50.0, datasets 2.21.0, accelerate 0.34.2, evaluate, matplotlib, gdown (+ their sub-dependencies).
- May use any publicly available pretrained LM, **as long as it was not itself trained on another intent/sales-intent dataset**. May use external datasets/resources for things like auxiliary pretraining, with two specifically-named exceptions that are banned outright (a specific GitHub benchmark repo and a specific HF model already trained on sales-intent data — not relevant to what we're asking below, just flagging that *some* specific resources are blocklisted, not that external resources are blocklisted in general).
- May not use the private/held-out test set's labels, directly or indirectly, for anything.

**Data**: 3000 labeled training examples (`train.jsonl`), each with a `language` field (`zh` or `en`), an `utterance`, optional prior dialogue `context` (list of turns), and 0+ of these 10 labels: `Ask_Price`, `Ask_Service`, `Ask_Spec`, `Compare_Competitor`, `Confirm_Order`, `Doubt`, `Need_More_Evidence`, `Need_Time_To_Think`, `State_Need`, `Too_Expensive`.

## Terminology (please read carefully — some of these are *our own* internal terms, not standard/assignment terms, to avoid ambiguity)

- **`train_split`**: 2550 of the 3000 training examples — what models actually train on.
- **`dev`**: **our own internal validation split** — the other 450 of the 3000 training examples, held out by us (not provided by the assignment) so we can tune per-class decision thresholds and compare configurations *without* ever touching the real test labels. Roughly matches the full training set's language ratio (~75% zh / ~25% en).
- **`public test`**: an assignment-provided 500-example set whose gold labels are *also* released to us (unusual — meant as a self-check/diagnostic). Its language ratio is close to 50/50 zh/en, unlike `train_split`/`dev`'s 75/25 — this mismatch matters, see below.
- **private test set**: the real held-out grading set. We never see it or its labels.
- **zh route / en route**: our architecture routes each example to a *separate* fine-tuned model per language (using the given `language` field, no language detection needed) rather than training one bilingual model. "The en route" = whichever model is currently assigned to handle English examples.

## Current status

### Architecture
A **language-routed ensemble**: one fine-tuned encoder for Chinese examples, one for English examples, routed by the given `language` field. This beat every single bilingual model tried by a wide margin (~+0.10 Macro-F1). Dialogue context is **dropped entirely** from the input (just the current utterance) — this outperformed every context-inclusion strategy tried (last-2-turns, last-4-turns, naive truncation, head+tail token-aware truncation), on both languages, at every training-data scale tried. This was a genuine surprise relative to round 1's assumption that context would help.

### What's been tried (condensed — happy to share the full ~750-line experiment log if useful)
- **Models tried as either route**: `bert-base-multilingual-cased`, `xlm-roberta-base`, `hfl/chinese-macbert-base`, `roberta-base`, `SpanBERT/spanbert-base-cased`, `microsoft/deberta-v3-base`, `microsoft/mdeberta-v3-base`, `BAAI/bge-m3`, `Qwen/Qwen2.5-0.5B-Instruct` (fine-tuned as a classifier via `AutoModelForSequenceClassification`, not prompted). **`BAAI/bge-m3`** (568M params, originally a multilingual retrieval/embedding model) unexpectedly won both routes on the first try, beating every purpose-built classification encoder.
- **Loss functions**: plain BCE, weighted BCE (`pos_weight = sqrt((N-N_k)/N_k)`), Asymmetric Loss (Ridnik et al. 2021). ASL won clearly on the English route; all three were within noise on the Chinese route (which has ~3x more training data).
- **Learning rate**: turned out to matter enormously for the English route specifically (627 training examples) — 2e-5 was clearly suboptimal (dev Macro-F1 ~0.75 at flat 0.5 threshold), 1e-4/5e-5 much better (~0.79). The Chinese route (~1900 examples) was much less LR-sensitive. This wasn't anticipated going in — round 1's advice didn't flag LR search as a priority lever, but it turned out to be the single biggest gain found after the initial architecture/context findings.
- **Epoch budget**: every model needed far more epochs than seemed obvious from an early "looks plateaued" read of the loss curve — this was independently discovered and re-discovered twice (for XLM-R, then for mBERT) before we learned to just always test more epochs rather than trust an apparent plateau.
- **Per-class threshold tuning** (vs. flat 0.5): consistently the single largest, cheapest lever — often +0.05 to +0.15 Macro-F1, especially on rare classes.
- **Ensembling already-trained checkpoints** (simple logit averaging): helped consistently on the Chinese route; genuinely ambiguous on the English route — improved our internal `dev` score but sometimes *hurt* the `public test` score, which we now attribute to `dev`'s small English slice (~117 examples) being too noisy to trust for this kind of comparison.
- **Domain-adaptive MLM pretraining**: continued unsupervised MLM pretraining of `roberta-base` on ~128K unlabeled English utterances from MultiWOZ (task-oriented booking dialogues — hotels/restaurants/taxis), no labels used at all, then fine-tuned as usual on the English route. **Negative result** — no better than the vanilla (non-domain-adapted) model at the same learning rate, possibly slightly worse.
- **Back-translation augmentation**: round-trip machine translation (English→Chinese→English via small MarianMT models) to generate same-label paraphrases, doubling the English route's training set from 627 to 1254. **Negative result, twice** — worse than no augmentation at two different learning rates tried.

### Current best numbers

| Config | Public test (Macro-F1 / Micro-F1) | `dev` (Macro-F1 / Micro-F1) |
|---|---|---|
| Simplest candidate (single checkpoint per route, no ensembling) | 0.8057 / 0.8238 | 0.8502 / 0.8556 |
| Ensembled + LR-tuned candidate | 0.8218 / 0.8316 | 0.8335 / 0.8442 |
| **BGE-M3, single checkpoint per route (current best)** | **0.8260 / 0.8273** | **0.8605 / 0.8618** |

Grading tiers: Basic 0.62/0.73 (cleared by all three), Medium 0.76/0.79 (cleared by all three), **Challenge 0.82/0.83** — cleared on `dev` by the BGE-M3 candidate, not yet reliably cleared on `public test`.

### The concrete bottleneck
The **English route is weaker than the Chinese route** on every model tried (Chinese route ~0.86–0.87 tuned Macro-F1 on `dev`; English route ~0.83–0.84 at best). This is plausibly a **small-data problem**: only 627 English training examples vs. ~1900 Chinese. Two of the ten labels — `Too_Expensive` and `Compare_Competitor` — remain the weakest classes specifically on the English route (both are also globally the rarest labels, ~6–7% prevalence). Two techniques aimed at exactly this bottleneck (domain-adaptive pretraining, back-translation augmentation) both failed to help, which suggests either the wrong technique or the wrong specific approach/corpus was tried, not that nothing can help.

## What we want this round to dig into (this is the actual ask)

Round 1 was a good general survey but stayed at the level of "here's a reasonable model shortlist and loss function." This round should be a deeper, more specific literature dive into methods for the *specific* situation we're now in: **a good multilingual/embedding-model architecture already found, with a small-labeled-data bottleneck on one language that resists two standard remedies.** Please look into:

1. **Few-shot / small-N text classification fine-tuning literature specifically** (not general text classification) — e.g. SetFit and other contrastive/sentence-embedding-based few-shot fine-tuning approaches, Pattern-Exploiting Training (PET/iPET), prompt-based fine-tuning for small-N classification. Given that our best-performing model (BGE-M3) is itself an embedding/retrieval model, is there a stronger paradigm than plain `AutoModelForSequenceClassification` fine-tuning for using an embedding model as a small-data classifier (e.g. contrastive fine-tuning on the embedding space directly, then a lightweight classifier head, rather than end-to-end fine-tuning with a randomly-initialized classification head)?

2. **The fine-tuning-instability-on-small-datasets literature** — there's a well-known body of work on BERT-family fine-tuning being unstable/sensitive to learning rate and seed specifically on small datasets (e.g. work following "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping" or "On the Stability of Fine-tuning BERT"-style papers). Given our own finding that LR mattered enormously for the 627-example English route, what does this literature recommend beyond "try several LRs" — e.g. specific warmup/schedule recipes, bias-correction fixes, layer-wise LR decay, or re-initializing the top few encoder layers before fine-tuning on small data?

3. **Cross-lingual data augmentation via *translate-train* (not back-translation)** — we tried back-translating our own small English set (paraphrasing existing examples). We have **not** tried translating our *abundant* Chinese training data (~1900 examples) into English to directly expand the English training set with genuinely new content (not just paraphrases of what we already have). Is translate-train a well-established, higher-value technique than back-translation for exactly this kind of resource-imbalanced bilingual setup? Any caveats (translation-artifact detectability, label transfer fidelity) we should know about before trying it?

4. **Techniques specifically for rare-class performance in multi-label settings**, beyond what we've already tried (Asymmetric Loss, weighted BCE, per-class threshold tuning) — e.g. Distribution-Balanced Loss (Wu et al.), Class-Balanced Loss (Cui et al.), any label-co-occurrence-aware approaches, or metric-learning/prototype-based heads for the specific rare classes (`Too_Expensive`, `Compare_Competitor`) that remain weak on the English route even after everything tried so far.

5. **A genuine SOTA/leaderboard search**: are there published results (papers, Kaggle/competition writeups, or production GitHub repos) for multi-label short-text or dialogue-intent classification under a similarly small-N, imbalanced, bilingual/cross-lingual setting that we should be benchmarking our approach against or borrowing tricks from? Round 1 named some candidate datasets (MultiWOZ, SGD, MASSIVE) but we're now specifically interested in *methods/papers*, not just datasets.

6. **HuggingFace model search, narrower and deeper than round 1**: given BGE-M3's surprising win, are there other multilingual sentence-embedding/retrieval models worth trying as classifiers under the same recipe — e.g. `intfloat/multilingual-e5-large`, `Alibaba-NLP/gte-multilingual-base`, `sentence-transformers/LaBSE`, or similar — that fit the 10GB VRAM / 4GB-download-budget-after-fp16 constraints (BGE-M3 fp16 is ~1.1GB per route; a same-size-class alternative would be similarly packageable)? Please include actual parameter counts and approximate fp16 disk size for each, not just names.

Please structure the response by these six numbered items, cite sources with links throughout, and end with a short prioritized list of what you'd actually try first given our compute/time budget (single RTX-class GPU class of hardware for dev iteration, each training run currently taking 1–5 minutes).
