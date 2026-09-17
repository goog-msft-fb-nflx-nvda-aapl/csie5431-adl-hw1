# Research brief: multi-label sales-intent classification (ADL HW1 / A1)

I'm doing homework 1 for a grad NLP course (multi-agent sales/customer game project). This assignment (A1) is BERT/transformer-based **multi-label intent classification** on bilingual (zh/en) sales-conversation utterances, with optional dialogue context. I want a literature + tooling survey before I commit to an approach. Please research and answer the questions below, citing sources (papers, HF model cards, GitHub repos) with links.

## Task recap

- 10 intent labels (multi-label, 0 to K per utterance): `Ask_Price`, `Ask_Service`, `Ask_Spec`, `Compare_Competitor`, `Confirm_Order`, `Doubt`, `Need_More_Evidence`, `Need_Time_To_Think`, `State_Need`, `Too_Expensive`.
- Input = current `utterance` + optional prior dialogue `context` (list of user/assistant turns, sometimes empty ~10% of the time).
- Metrics: Macro-F1 and Micro-F1, both must clear thresholds simultaneously (basic 0.62/0.73, medium 0.76/0.79, challenge 0.82/0.83).
- Bilingual: Chinese and English utterances in the same dataset, not separated by task.

## Hard constraints (rule these out anything that doesn't fit)

- **Compute at inference/grading time:** Ubuntu 20.04, 32GB RAM, **RTX 3080 Ti with 10GB VRAM**, 20GB disk, **no network access** after `download.sh` runs. `run.sh` must finish in 2 hours on the held-out test set.
- **Download budget:** `download.sh` may fetch at most **4GB total** (model weights + tokenizer + any data), must finish within 1 hour at ~10MB/s, from Dropbox/Google Drive only.
- **Allowed packages only:** Python 3.10/3.12, PyTorch 2.11.0, scikit-learn 1.9.0, tqdm, numpy, pandas, transformers 4.50.0, datasets 2.21.0, accelerate 0.34.2, evaluate, matplotlib, gdown (+ their sub-dependencies). No other libraries.
- **Model/data rules:**
  - May use external datasets/resources EXCEPT the "SalesLLM" benchmark (`Bairong-Xdynamics/Benchmarking-LLM-Realistic-Selling-Skill` on GitHub).
  - May use any publicly available pretrained LM (encoder or decoder), but not a model already trained on another intent/sales-intent dataset, e.g. `MultiSense/SaleIntent_bert` on HuggingFace is explicitly banned, and anything similar needs to be cleared first.
  - Cannot scrape/annotate more data from the original data source, cannot touch test labels.
- Report also needs an "Additional Exploration" section (2%) — an open slot for whatever finding is most interesting, so surface any promising side-avenue you find.

## What I already know from my own EDA (train.jsonl, 3000 examples; test.json, 500 public examples with gold labels released)

- **Label frequency (train):** Ask_Service 34.9%, Doubt 29.6%, Ask_Spec 28.3%, Need_More_Evidence 22.5%, Ask_Price 16.8%, Need_Time_To_Think 14.7%, Confirm_Order 14.5%, State_Need 9.5%, Compare_Competitor 6.7%, Too_Expensive 6.6%. Public test gold label rates are close to train's (within ~3.5 points per class), so imbalance shape looks stable across splits.
- **Cardinality:** every example has ≥1 label (no zero-label rows in train). Distribution over train: 1 label 40%, 2 labels 40%, 3 labels 17%, 4 labels 3%, 5+ labels <0.5%. Similar shape on public test.
- **Label co-occurrence:** top pairs are (Ask_Service, Doubt), (Ask_Spec, Doubt), (Ask_Service, Ask_Spec), (Doubt, Need_More_Evidence), (Ask_Service, Need_More_Evidence) — the frequent classes cluster together; the rare classes (Compare_Competitor, Too_Expensive, State_Need) co-occur less.
- **Language split — this looks like the "different distribution" the assignment warns about:** train is 75.2% zh / 24.8% en (2256/744), but the **public test set is ~48.6% zh / 51.4% en (243/257) — roughly balanced**, unlike train. So English is much more heavily represented at test time relative to train. Per-language label rates within train look similar between zh and en (no obvious per-class skew by language), so the shift looks like it's mainly in language *proportion*, not label semantics — but I haven't verified that holds on the private test set.
- **Context:** ~9.9% of train examples (~10.6% of test) have empty context (`[]`). Non-empty context caps at 4 turns in train (median 4). Some dialogues are long: full (context + utterance) character length has mean 912, median 599, p90 2192, p99 3956, max 7220 characters — a meaningful long tail that will exceed a 512-token BERT window if context is concatenated naively, especially combined with Chinese text (higher info density per token) vs English.
- Data is a mix of e-commerce-style and financial/insurance-style sales dialogues (from what I've read so far), both zh and en.

## Questions to research

### 1. EDA — what am I missing?
Given the above, what additional EDA would a strong practitioner do before modeling? E.g.: actual tokenizer-based length distributions (not char counts) for a couple of candidate zh/en tokenizers, per-class length/context-turn correlation, keyword/n-gram analysis per label (especially to disambiguate near-confusable pairs like `Doubt` vs `Need_More_Evidence` vs `Ask_Spec`, or `State_Need` vs `Ask_Spec`), label co-occurrence structure as a graph/correlation matrix, checking whether English vs Chinese utterances differ systematically in length or label mix beyond what I found, and any red flags for train/test distribution shift I should specifically check for (since the assignment explicitly warns train/test distributions may differ).

### 2. SOTA / practical model choices for bilingual short-text multi-label intent classification
Given the 10GB VRAM / 4GB download budget / offline inference / 2-hour batch inference constraints, survey encoder-based options first (this is explicitly framed as a "Transformer/BERT" assignment):
- Multilingual encoders that handle zh+en well: e.g. `bert-base-multilingual-cased`, `xlm-roberta-base`/`-large`, multilingual E5, `google/gemma` embedding variants, BGE-M3, etc.
- Chinese-specialized encoders (for zh-only ablations / "Chinese vs English" report question): `bert-base-chinese`, `hfl/chinese-roberta-wwm-ext`, `hfl/chinese-macbert-base/large`, ERNIE variants available on HF.
- English-specialized encoders for the same ablation: `bert-base-uncased`, `roberta-base`, `deberta-v3-base`.
- For each candidate: HF repo id, param count, approximate fp16/fp32 disk size, license, and whether it's known to perform well on multi-label short-text / intent / dialogue-act classification specifically (cite benchmarks if you can find them — e.g., performance on MultiWOZ dialogue-act tagging, Chinese/English GLUE-style intent tasks, or multi-label text classification leaderboards).
- Also note whether a small decoder LLM (e.g., Qwen2.5-0.5B/1.5B-Instruct, Gemma-2 2B, Llama-3.2-1B-Instruct) used as a zero/few-shot or lightly fine-tuned multi-label classifier is a credible alternative given the compute/latency budget, and how it tends to compare against fine-tuned BERT-style encoders for this class of task in the literature.

### 3. Public datasets that could legally help here
Besides my own train data, are there publicly available datasets on HuggingFace/GitHub that could be used for auxiliary pretraining, data augmentation, or transfer learning for this task — while respecting the rule that I can't use anything trained on another intent/sales-intent dataset as my *final* classifier, and can't use the specifically-banned SalesLLM benchmark? I'm interested in candidates like:
- Dialogue-act / multi-label conversational-intent datasets (e.g., MultiWOZ dialogue acts, Schema-Guided Dialogue (SGD), other multi-intent SLU datasets).
- Chinese e-commerce / customer-service dialogue corpora (e.g., JDDC, ECD, or similar corpora sometimes referenced in Chinese customer-service NLP papers).
- General bilingual/multilingual intent or sentiment/dialogue corpora that could help with pretraining a better zh/en shared representation, or that could be used to build a synthetic augmentation pipeline (paraphrase/back-translation) that stays within the allowed-package list.
For each: what it is, label schema, size, license, HF/GitHub link, and an honest read on whether it's actually close enough to this task's label schema to be useful (versus just noise).

### 4. Long-input handling strategies
Given the 512-token typical BERT context window vs. our tail of up to ~7200-character dialogues, what are the standard practical strategies and their tradeoffs, specifically for a "does dialogue context help, and how do you handle long ones" report question? E.g.: truncate-from-head vs truncate-from-tail vs head+tail concatenation, keep-last-K-turns only, hierarchical/turn-level encoding (encode each turn, pool), long-context encoder checkpoints (Longformer/BigBird zh or multilingual variants if they exist), or simple context-summarization via truncation heuristics. Note which classes of intent (from ours) are more likely to depend on long context vs. being decidable from the current utterance alone, if literature/practice has a view on this.

### 5. Imbalanced multi-label training & macro-F1 optimization
What's current best practice for training multi-label classifiers under label imbalance like ours (rarest class ~6.6%, most common ~34.9%) when macro-F1 is a hard grading gate? E.g.: BCE-with-class-weights vs focal loss vs asymmetric loss (ASL) for multi-label, per-class decision threshold tuning (vs a flat 0.5) optimized on a held-out split, oversampling/undersampling approaches adapted to multi-label (MLSMOTE or similar), and any pitfalls specific to optimizing macro-F1 vs micro-F1 simultaneously (since both are graded and can trade off against each other).

### 6. Robustness to the train/test language-ratio shift
Given train is ~75/25 zh/en but public test is ~49/51 zh/en, what does the literature/practice say about training a bilingual multi-label classifier that's robust to this kind of shift — e.g., language-balanced batch sampling or re-weighting during fine-tuning, choosing a genuinely multilingual encoder over concatenating monolingual ones, or other techniques? Also flag if you think my read of the "different distribution" hint might mean something else (label distribution shift, not just language ratio) that I should check for instead/also.

Please structure the answer by these six numbered sections, and end with a short prioritized recommendation (what you'd actually try first given the constraints) plus a list of anything you're materially unsure about that I should double check myself.
