# 1. EDA to add

Your EDA already identifies the two highest-risk issues: **multi-label dependence** and **language-proportion shift**. Before modeling, I would add the following.

## Tokenizer-based lengths

Measure lengths with the actual tokenizer for each candidate checkpoint, separately for:

- Current utterance only.
- Context only.
- Context plus utterance.
- Chinese, English, and mixed-language examples.
- Empty versus non-empty context.
- Each label and each label pair.

Report median, p90, p95, p99, maximum, and the fraction exceeding 128, 256, 384, and 512 tokens. Character length is not a sufficient proxy: WordPiece, SentencePiece, and Chinese character segmentation produce materially different sequence lengths.

Good tokenizer comparisons include:

- `google-bert/bert-base-multilingual-cased`.
- `FacebookAI/xlm-roberta-base`.
- `hfl/chinese-macbert-base` for Chinese ablations.
- `microsoft/mdeberta-v3-base` as a multilingual alternative.

Also measure **token inflation**:

\[
\text{inflation} =
\frac{\text{tokens(context + utterance)}}{\text{tokens(utterance)}}
\]

This will tell you whether context is adding useful evidence or mostly consuming the window.

## Split and leakage checks

Do not rely only on aggregate label rates. Check:

- Exact duplicate utterances across train and public test.
- Near duplicates after whitespace, punctuation, case, and Chinese normalization.
- Repeated dialogue prefixes or conversation IDs.
- Product, insurer, bank, merchant, and domain distributions.
- Named entities, prices, currencies, URLs, phone numbers, and product IDs.
- Label cardinality by language and by domain.
- Co-occurrence distributions by language.
- Whether examples from the same dialogue occur in both train and validation.
- Whether English examples are translations or paraphrases of Chinese examples.
- Whether public-test examples are lexically closer to train English than train Chinese.

A random split can be over-optimistic if adjacent turns from one dialogue appear in both partitions. If dialogue IDs are available, use a **grouped split by dialogue**.

For train/test shift, compare:

- Character n-grams and tokenizer token frequencies.
- Utterance length and context length.
- Label cardinality.
- Per-label prevalence.
- Pairwise label co-occurrence.
- Domain/product vocabulary.
- Language-specific distributions.
- Embedding distributions, using a simple classifier trained to distinguish train from public test.

A train-versus-test discriminator with high AUC is a useful warning even when label frequencies look stable.

## Label-specific linguistic analysis

For every label, compute:

- Positive versus negative TF-IDF unigrams, bigrams, and character n-grams.
- Log-odds ratios with smoothing.
- Precision of individual lexical cues.
- Cue overlap between confusing labels.
- Example-based nearest neighbors using TF-IDF or encoder embeddings.

Particular distinctions worth manually auditing:

| Confusion | Useful diagnostic |
|---|---|
| `Doubt` vs `Need_More_Evidence` | Is the speaker expressing skepticism, or explicitly requesting proof, documents, demonstrations, or evidence? |
| `Ask_Spec` vs `State_Need` | Is the utterance asking about a product attribute, or describing a desired requirement? |
| `Too_Expensive` vs `Ask_Price` | Does it merely request a price, or evaluate the price negatively? |
| `Need_Time_To_Think` vs `Doubt` | Is there explicit postponement or indecision, versus concern or skepticism? |
| `Compare_Competitor` vs `Ask_Spec` | Is another product/provider explicitly used as the comparison anchor? |
| `Confirm_Order` vs `State_Need` | Is there commitment to purchase, or only a statement of requirements? |

For multi-label data, calculate **conditional label rates** such as:

\[
P(y_j=1 \mid y_i=1)
\]

and compare them with \(P(y_j=1)\). A heatmap of lift,

\[
\operatorname{lift}(i,j)
=
\frac{P(y_i=1,y_j=1)}
{P(y_i=1)P(y_j=1)},
\]

is more informative than raw co-occurrence counts.

## Language analysis

Your conclusion that the shift is mainly a language-ratio shift is plausible, but verify:

- Label rates conditional on language.
- Cardinality conditional on language.
- Token length conditional on language.
- Context availability conditional on language.
- Context-turn count conditional on language.
- Domain conditional on language.
- The same statistics for public test.
- Interaction effects such as `language × label`, `language × context`, and `language × cardinality`.

A useful diagnostic is to train separate lightweight classifiers for Chinese and English, then compare:

1. A multilingual model trained normally.
2. The same model with language-balanced sampling.
3. Separate monolingual models.
4. A multilingual model with a language indicator.

This makes the report’s “Chinese versus English” claim empirical rather than anecdotal.

# 2. Practical model choices

## Recommended encoder candidates

Approximate sizes below refer to parameter storage only. Actual repositories may include optimizer files, safetensors metadata, tokenizer files, and multiple weight formats.

| Model | HF ID | Params | Approx. fp16 / fp32 weights | License | Assessment |
|---|---|---:|---:|---|---|
| mBERT | `google-bert/bert-base-multilingual-cased` | ~178M | ~0.36 / 0.71 GB | Apache-2.0 | Strong, simple bilingual baseline; supports 104 languages.  [huggingface](https://huggingface.co/google-bert/bert-base-multilingual-cased) |
| XLM-R base | `FacebookAI/xlm-roberta-base` | ~270M | ~0.54 / 1.08 GB | MIT-style model release; verify repository terms | My default multilingual candidate; pretrained on 100 languages and CommonCrawl.  [huggingface](https://huggingface.co/FacebookAI/xlm-roberta-base) |
| XLM-R large | `FacebookAI/xlm-roberta-large` | ~550M | ~1.1 / 2.2 GB | Check exact model card | Potentially stronger, but unnecessary risk with 10 GB VRAM and only 3,000 examples. |
| mDeBERTa-v3 base | `microsoft/mdeberta-v3-base` | ~280M | ~0.56 / 1.12 GB | MIT | Strong multilingual alternative; 250K vocabulary and 512-token limit.  [huggingface](https://huggingface.co/microsoft/mdeberta-v3-base) |
| DistilmBERT multilingual | `distilbert/distilbert-base-multilingual-cased` | 134M | ~0.27 / 0.54 GB | Apache-2.0 | Faster and smaller; a useful speed baseline, but usually less accurate than full mBERT.  [huggingface](https://huggingface.co/distilbert/distilbert-base-multilingual-cased) |
| BGE-M3 | `BAAI/bge-m3` | ~568M | ~1.14 / 2.27 GB | MIT | Multilingual, 8,192-token embedding model; attractive for retrieval or frozen-feature experiments, not my first supervised classifier.  [build.nvidia](https://build.nvidia.com/baai/bge-m3.md) |
| Chinese BERT | `google-bert/bert-base-chinese` | ~110M | ~0.22 / 0.44 GB | Apache-2.0 | Appropriate Chinese-only ablation; weaker choice for the joint task. |
| Chinese RoBERTa | `hfl/chinese-roberta-wwm-ext` | ~102M | ~0.20 / 0.41 GB | Check HF card | Strong Chinese baseline; cannot directly handle English as well as multilingual models. |
| Chinese MacBERT | `hfl/chinese-macbert-base` | ~102M | ~0.20 / 0.41 GB | Apache-2.0 according to published catalog metadata | Very reasonable Chinese ablation.  [ai.azure](https://ai.azure.com/catalog/models/hfl-chinese-macbert-base) |
| English BERT | `google-bert/bert-base-uncased` | 110M | ~0.22 / 0.44 GB | Apache-2.0 | Useful English-only ablation. |
| RoBERTa base | `FacebookAI/roberta-base` | 125M | ~0.25 / 0.50 GB | MIT | Strong English encoder; not suitable as the sole bilingual model. |
| DeBERTa-v3 base | `microsoft/deberta-v3-base` | ~184M total, including embeddings | ~0.37 / 0.74 GB | MIT | Strong English baseline; the model card reports strong MNLI and SQuAD results, but it is English-focused.  [huggingface](https://huggingface.co/microsoft/deberta-v3-base) |

The exact disk size should be checked at the pinned revision you download. In particular, do not assume that the nominal parameter count equals the download size.

### My shortlist

1. `FacebookAI/xlm-roberta-base`.
2. `microsoft/mdeberta-v3-base`.
3. `google-bert/bert-base-multilingual-cased`.
4. `hfl/chinese-macbert-base` and `roberta-base` only as language-specific ablations.
5. `BAAI/bge-m3` only if the 8K context capability is worth a separate experiment.

XLM-R is pretrained over 100 languages and is a natural fit for jointly encoding Chinese and English.  mDeBERTa has a large multilingual vocabulary and is another reasonable candidate, but it is heavier than XLM-R base in embedding parameters. [huggingface](https://huggingface.co/FacebookAI/xlm-roberta-base)

There is no reliable basis for claiming that any of these generic checkpoints is specifically optimal for your ten labels. Intent-classification performance depends heavily on fine-tuning, threshold calibration, and split construction. Treat benchmark claims from unrelated intent datasets as directional, not predictive of your assignment score.

## Decoder alternatives

Potential candidates include:

- `Qwen/Qwen2.5-0.5B-Instruct`: 0.49B parameters, 32,768-token context, Apache-2.0. [huggingface](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4)
- `Qwen/Qwen2.5-1.5B-Instruct`: likely still within the storage budget, but materially slower and more difficult to fine-tune.
- Gemma 2B instruction models: gated Google Gemma terms and therefore less convenient for an offline reproducible assignment. [huggingface](https://huggingface.co/google/gemma-2b)
- Llama 3.2 1B Instruct: custom Llama Community License, which requires careful license checking. [huggingface](https://huggingface.co/ModelCloud/Llama3.2-1B-Instruct)

A decoder can be useful for zero-shot or few-shot exploration, especially because your labels have semantic names. However, for 3,000 labeled examples and a strict F1 gate, a fine-tuned encoder classifier is more credible:

- Encoders produce one vector and ten logits in a single forward pass.
- Multi-label BCE training is straightforward.
- Per-label thresholds are easy to tune.
- Output formatting errors are eliminated.
- Inference is faster and more deterministic.
- A 0.5B decoder may understand the task but still produce invalid, incomplete, or inconsistently formatted labels.

I would not make a decoder the main submission unless a small controlled experiment shows a substantial gain. It is better suited to the Additional Exploration section.

# 3. Public datasets and auxiliary resources

The most useful resources are those that transfer conversational structure rather than superficial sales vocabulary.

| Dataset | Scale and schema | License/link | Usefulness |
|---|---|---|---|
| MultiWOZ 2.2 | 8,438 train, 1,000 validation, 1,000 test dialogues; 42,190 train turns; dialogue acts, slots, and domains | Apache-2.0 HF release.  [huggingface](https://huggingface.co/datasets/Abhinav242004/multi_woz_v22) | Useful for dialogue-act or context-aware auxiliary training, but English-only and task-oriented rather than sales-oriented. |
| Schema-Guided Dialogue | 16,142 train dialogues, 2,482 validation dialogues, 4,201 test dialogues; 16–18 domains, intents, slots, dialogue acts | CC BY-SA 4.0.  [github](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Strong for context dependence and request/confirm/deny behavior; label mapping to your ten classes is indirect. |
| MASSIVE | More than 1M utterances, 51 languages, 18 domains, 60 intents, 55 slots | CC BY 4.0 according to Amazon’s release.  [arxiv](https://arxiv.org/abs/2204.08582) | Best multilingual transfer candidate, but it is single-turn virtual-assistant data and mostly single-intent. Use for representation transfer, not label remapping. |
| SLURP | English spoken-language NLU dataset with intents and slots | Check the source repository’s exact license | Useful for intent paraphrase behavior, less useful for sales semantics. |
| CLINC OOS | English banking/general assistant intents and out-of-scope examples | Check the exact HF dataset card | Useful for out-of-scope and intent-boundary experiments, not direct label transfer. |
| JDDC | Chinese customer-service dialogues, generally centered on e-commerce/customer support | Verify the exact GitHub/HF release and license before use | Potentially close in conversational style, but licensing and schema vary across mirrors. |
| ECD and related Chinese customer-service corpora | Chinese e-commerce conversations with task- or response-oriented annotations in some releases | Verify source-specific terms | Potentially valuable for unsupervised domain-adaptive pretraining, but not automatically legal or consistently labeled. |

### Best legal/use strategy

Use public datasets in one of three conservative ways:

1. **Unsupervised domain-adaptive pretraining**  
   Continue masked-language-model training on permitted unlabeled Chinese/English dialogue text, then fine-tune only on your assignment labels.

2. **Auxiliary dialogue-act training**  
   Train a shared encoder on MultiWOZ or SGD dialogue-act labels, discard the auxiliary head, and fine-tune on the ten assignment labels.

3. **Representation or augmentation experiments**  
   Use MASSIVE or permitted bilingual corpora to study paraphrase robustness, but do not treat their intent labels as equivalent to your labels.

For strict compliance, keep a table containing source URL, revision/hash, license text, data fields used, and whether labels influenced the final classifier. The banned SalesLLM benchmark should not appear in the pipeline at all.

### Synthetic augmentation

Given the allowed packages, realistic options are limited:

- Rule-based paraphrase templates.
- Chinese punctuation and whitespace normalization.
- Controlled price, product, and provider substitutions.
- English capitalization and contraction variants.
- Back-translation only if you already have an allowed offline translation model, which may violate the 4 GB budget and is not naturally available from the listed packages.
- Context perturbations such as dropping irrelevant turns or adding neutral turns.

Avoid generating synthetic labels by simply mapping MASSIVE or MultiWOZ intents to your labels. That creates semantic label noise, especially for `Doubt`, `Need_More_Evidence`, `State_Need`, and `Ask_Spec`.

# 4. Long-input handling

A standard BERT-style model has a 512-token input limit. Longformer uses local and global sparse attention to process thousands of tokens, while BigBird uses block-sparse attention and supports sequences up to approximately 4,096 tokens in its standard checkpoint. [huggingface](https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/longformer)

However, long-context models are not automatically the best choice here. Your dataset contains only 3,000 labeled examples, and the current utterance is probably much more predictive than the entire conversation. I would evaluate context strategies incrementally.

## Recommended ablation ladder

| Variant | Construction | Expected behavior |
|---|---|---|
| U | Current utterance only | Essential baseline; likely strong for explicit price, specification, comparison, and ordering cues. |
| L4 | Current utterance plus last four turns | Natural baseline because train dialogues cap at four context turns. |
| Last-\(K\) tokens | Preserve the most recent context tokens plus full current utterance | Protects local conversational state. |
| Head + tail | Keep early context summary plus recent turns | Preserves initial product/topic setup and current state. |
| Turn-level | Encode each turn separately, pool turn representations, classify | More principled, but more code and compute. |
| Long-context checkpoint | BigBird or Longformer | Worth testing only if truncation ablations show a clear loss. |

### Truncation policy

For sales conversations, **preserve the current utterance in full**. Then allocate remaining tokens to recent context. A practical format is:

```text
[CTX_OLD] ... [CTX_RECENT] ... [UTTERANCE] current utterance
```

Use explicit speaker and turn markers such as:

```text
[USER] ...
[ASSISTANT] ...
[USER] ...
[CURRENT_USER] ...
```

Compare:

- Last-context-first truncation.
- Head-plus-tail context.
- Last two turns only.
- Full current utterance plus a fixed 128–256 tokens of context.

Do not silently truncate the current utterance. A price, competitor name, or negation near the end can determine the label.

## Which labels need context?

Likely context-sensitive:

- `Confirm_Order`: may depend on what product, plan, or option is being confirmed.
- `State_Need`: the requirement may be elliptical.
- `Doubt`: “I’m still not convinced” requires prior claims.
- `Need_More_Evidence`: the requested evidence may refer to a previous salesperson statement.
- `Need_Time_To_Think`: interpretation can depend on whether an offer or recommendation was just made.
- `Compare_Competitor`: the comparison target may appear earlier.

More often utterance-decidable:

- `Ask_Price`, if a price request is explicit.
- `Ask_Service`, if the service is directly named.
- `Ask_Spec`, if the requested attribute is explicit.
- `Too_Expensive`, if explicit price dissatisfaction is present.

These are hypotheses, not guarantees. Test them using paired evaluation:

- Same utterance with context removed.
- Same utterance with context retained.
- Examples grouped by label and context length.
- Performance change by language.

A particularly informative diagnostic is the **context gain**:

\[
\Delta F1_c =
F1_c(\text{utterance + context})
-
F1_c(\text{utterance only})
\]

Report this per label rather than only overall Macro-F1.

# 5. Imbalance and macro-F1

Your imbalance is moderate rather than extreme: the rarest label is approximately 6.6%, while the most frequent is approximately 34.9%. The larger issue is that every example has at least one positive label, so each training row contributes nine or more negative label decisions.

## Start with BCE

Use logits and `BCEWithLogitsLoss`. First establish:

- Unweighted BCE.
- Positive-class weighting.
- Per-label threshold tuning.

For label \(k\), a conventional positive weight is:

\[
w_k =
\frac{N-N_k}{N_k}.
\]

But applying the full inverse-frequency weight can overcorrect and damage Micro-F1. Test a tempered version such as:

\[
w_k =
\sqrt{\frac{N-N_k}{N_k}}
\]

or interpolate between 1 and the inverse-frequency weight.

## ASL and focal loss

Asymmetric Loss explicitly treats positive and negative examples differently, downweights easy negatives, and focuses on hard cases; the authors provide an implementation. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)

This matches your setting better than ordinary focal loss because the negative-label population dominates. A reasonable experiment is:

- \( \gamma_+ = 0 \)
- \( \gamma_- \in \{2,4\} \)
- Probability margin \(m \in \{0,0.05\} \)

Do not assume ASL will beat BCE on only 3,000 examples. Tune it against repeated validation splits.

Focal loss can help, but it adds another focusing hyperparameter and can reduce calibration. I would try it after BCE and ASL.

## Threshold tuning is essential

A flat threshold of 0.5 is rarely optimal for macro-F1. Tune one threshold \(t_k\) per label on validation predictions:

\[
\hat y_k = \mathbf{1}[p_k \geq t_k].
\]

Search a grid such as 0.05–0.95 and maximize a validation objective that reflects the grading gate. Options include:

- Direct Macro-F1.
- \(0.5 \times \text{Macro-F1} + 0.5 \times \text{Micro-F1}\).
- A constrained objective that heavily penalizes either metric falling below its desired threshold.

Because your examples always have at least one label, consider a fallback:

- If no class exceeds its threshold, predict the highest-probability class.
- Do not force a fixed number of labels globally; cardinality varies from one to five or more.
- Optionally use a validation-derived cardinality prior as a secondary calibration signal.

Tune thresholds on a validation split only. Using public-test labels repeatedly for threshold selection is test leakage, even if those labels are released.

## Oversampling

Naïve row oversampling can distort co-occurrence patterns. An example containing `Too_Expensive` may also contain `Doubt`, so oversampling it changes both labels.

MLSMOTE was specifically proposed for imbalanced multi-label datasets and synthesizes examples using minority-label neighborhoods.  However, implementing it correctly for text requires a meaningful feature space and can create unrealistic text-label combinations. [simidat.ujaen](https://simidat.ujaen.es/en/bibcite/reference/512)

For this assignment I would prefer:

1. Stratified iterative multi-label splitting.
2. Loss weighting or ASL.
3. Threshold calibration.
4. Optional minority-focused batch sampling.
5. Only then, carefully audited augmentation.

## Macro/Micro tradeoff

Macro-F1 gives each label equal importance, so improving `Compare_Competitor` can matter as much as improving `Ask_Service`. Micro-F1 aggregates decisions and is dominated by common labels.

Monitor at least:

- Per-label precision, recall, and F1.
- Macro-F1.
- Micro-F1.
- Label cardinality of predictions.
- Predicted prevalence versus validation prevalence.
- Exact-match accuracy, as an additional diagnostic.
- Performance by language.

A model predicting too many labels can improve recall and sometimes Macro-F1 while damaging precision and Micro-F1. Threshold tuning must therefore optimize both metrics jointly.

# 6. Robustness to language-ratio shift

Your observed shift from 75.2% Chinese / 24.8% English in train to approximately 48.6% Chinese / 51.4% English in public test is a genuine covariate shift if the private test follows it. The fact that conditional label rates appear similar is encouraging: it suggests \(P(y\mid x,\text{language})\) may be relatively stable while \(P(\text{language})\) changes.

## Practical response

### Use balanced language sampling

During fine-tuning, construct batches with approximately equal Chinese and English examples. This prevents the English representation from being under-trained and makes the empirical training objective closer to the public-test mixture.

Do not discard data. Use either:

- A weighted sampler.
- Per-example loss weights.
- Balanced mini-batches.
- Oversampling of English examples.

With only 744 English training examples, oversampling can increase variance. Compare balanced sampling against ordinary sampling over several random seeds.

### Evaluate language-specific metrics

Report:

- Macro-F1 and Micro-F1 for Chinese.
- Macro-F1 and Micro-F1 for English.
- Overall metrics.
- Per-label metrics by language.
- Performance on context-empty versus context-present examples.

An overall score can hide a large English degradation after the test distribution becomes English-heavy.

### Prefer one genuinely multilingual encoder

For the main model, use XLM-R base, mBERT, or mDeBERTa. A shared multilingual encoder allows transfer between languages and keeps the classifier head aligned.

Do not concatenate separate Chinese and English models unless an ablation demonstrates a substantial gain. Two encoders increase memory, implementation complexity, and failure modes, and make shared label calibration harder.

### Add explicit language information only if useful

You can add a small language embedding or a language token, but first test whether the tokenizer/model already separates the languages reliably. A language token can help calibration but can also encourage language-specific shortcuts.

### Check whether the hint means more than language ratio

The assignment’s “different distribution” warning may refer to:

- Label prevalence shift.
- Label cardinality shift.
- Context availability shift.
- Dialogue-domain shift.
- Product/provider vocabulary shift.
- Different proportions of short and long examples.
- Different proportions of English examples with empty context.
- Different co-occurrence patterns.
- Hidden source or template changes.

Because you have public-test labels, calculate all of these directly for the public test, but select any final hyperparameters using a validation split that does not repeatedly absorb test information. Use the public test primarily as a diagnostic, not as a training or threshold-tuning set.

# Prioritized recommendation

1. **Build an utterance-only XLM-R-base baseline** with ordinary BCE, iterative multi-label validation splitting, and per-label threshold tuning.
2. Add **last-four-turn context**, preserving the current utterance in full; compare against utterance-only with paired per-label and per-language metrics.
3. Compare ordinary versus language-balanced sampling.
4. Try ASL and tempered positive weighting only after the baseline is stable.
5. Tune thresholds against a joint Macro/Micro objective, then freeze them.
6. Run Chinese-only MacBERT and English-only RoBERTa as ablations, not as the main bilingual solution.
7. Use the Additional Exploration section for either:
   - context-gain analysis by label, or
   - a small Qwen2.5-0.5B structured-output experiment, clearly labeled exploratory.

The 0.5B Qwen model is small enough to fit the hardware and has an Apache-2.0 model card, but I would expect a fine-tuned encoder to be more reliable for this ten-logit multi-label problem. [huggingface](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4)

## Material uncertainties to verify

- The exact licenses and revisions of `hfl/chinese-roberta-wwm-ext`, `hfl/chinese-macbert-base`, XLM-R, and every auxiliary dataset you actually download.
- Whether the private test follows the public test’s approximately balanced language ratio.
- Whether train and test contain dialogue-level or template-level leakage.
- Whether public-test labels may be used for model selection under your course’s rules.
- Whether the assignment’s “different distribution” hint refers to language proportion, domain/source mixture, or label/cardinality shift.
- Whether external auxiliary pretraining is permitted when its labels are not used, and whether continued MLM pretraining counts as using an external resource.
- Whether `transformers 4.50.0` supports the exact checkpoint architecture and tokenizer without additional packages.
- Whether the course environment permits downloading a checkpoint whose model card has gated access or nonstandard terms.
- Whether your validation split is sufficiently stable for per-label threshold tuning, particularly for `Compare_Competitor` and `Too_Expensive`.
- Whether BGE-M3’s 8K sequence support is worth its larger memory and training cost compared with simple turn truncation.