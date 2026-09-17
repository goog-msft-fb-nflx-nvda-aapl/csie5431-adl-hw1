# BERT Variants
**Applied Deep Learning** | September 8th, 2025  
[http://adl.miulab.tw](http://adl.miulab.tw)

---

## Three Types of Model Pre-Training
- **Encoder**: Bidirectional context. (Examples: BERT and its variants)
- **Decoder**: Language modeling; better for generation.
- **Encoder-Decoder**: Sequence-to-sequence model.

---

## Beyond BERT
To achieve **Better Performance** and **Wide Applications**, several BERT variants have been developed:
- XLNet
- RoBERTa
- SpanBERT
- XLM
- Multilingual BERT

---

## Transformer-XL *(Dai et al., 2019)*

### The Issue: Context Fragmentation
- **Long dependency**: Unable to model dependencies longer than a fixed length.
- **Inefficient optimization**: Ignores sentence boundaries.

### The Idea: Segment-Level Recurrence
- Previous segment embeddings are fixed and cached to be reused when training the next segment.
- Increases the largest dependency length by $N$ times ($N$: network depth).
- Resolves the context fragmentation issue and makes the dependency longer.

### Positional Encoding
- **Issue**: Naively applying segment-level recurrence doesn't work because absolute positional encodings are incoherent when reusing (e.g., `[0, 1, 2, 3] [0, 1, 2, 3]`).
- **Solution**: Use **relative positional encoding** to support state reuse, alongside learnable positional embeddings.

### Contributions
- **Longer context dependency**:
  - Learns dependencies longer than vanilla Transformers.
  - Better perplexity on long sequences.
  - Better perplexity on short sequences by addressing the fragmentation issue.
- **Speed increase**:
  - Processes new segments without recomputation.
  - Achieves up to 1,800+ times faster than a vanilla Transformer during evaluation on Language Modeling (LM) tasks.

---

## XLNet *(Yang et al., 2019)*

### Auto-Regressive (AR) vs. Auto-Encoding (AE)
- **Auto-Regressive (AR)**: Objective is modeling information based on either previous or following contexts.
- **Auto-Encoding (AE)**: Objective is reconstructing $\bar{x}$ from $\hat{x}$ (e.g., dimension reduction or denoising via Masked LM). Randomly masks 15% of tokens.
- **Issues with AE**:
  - *Independence assumption*: Ignores the dependency between masks.
  - *Input noise*: Discrepancy between pre-training and fine-tuning (with `[MASK]` vs. without `[MASK]`).

### Permutation Language Model
- **Goal**: Use AR and bidirectional contexts for prediction.
- **Idea**: Parameters shared across all factorization orders in expectation.
  - $T!$ different orders to a valid AR factorization for a sequence of length $T$.
  - Pre-training on sequences sampled from all possible permutations.
- **Implementation**: Only permute the factorization order.
  - Retains original positional encoding.
  - Relies on proper attention masks in Transformers.
  - *Resolves independence assumption and pretrain-finetune discrepancy issues.*

### Two-Stream Self-Attention
- **Content stream**: Predict other tokens.
- **Query stream**: Predict the current token.

### Contributions
- Uses **AR** for addressing the independence assumption.
- Uses **AE** for addressing the pretrain-finetune discrepancy.

---

## RoBERTa: Robustly Optimized BERT Approach *(Liu et al., 2019)*

### What's More in RoBERTa?
- **Dynamic masking**:
  - 10 different masking ways over 40 epochs.
  - *(BERT used static masking by preprocessing).*
- **Optimization**:
  - Peak learning-rate & warmup-steps tuned separately.
  - Large batch size (batch size = 8K).
- **Data**:
  - Trains only with full-length sequences. *(BERT used reduced length).*
  - Datasets: BookCorpus + English Wikipedia (16G), CC-News (76G), OpenWebText (38G), Stories (31G).

---

## SpanBERT *(Joshi et al., 2019)*

### Key Features
- **Span masking**: A random process to mask spans of tokens.
- **Single sentence training**: Uses a single contiguous segment of text for each training sample (instead of two).
- **Span boundary objective (SBO)**: Predict the entire masked span using only the span's boundary.

### Results
- Improvements noted in masking scheme and auxiliary objective.
- Particularly better for QA, NLI, and coreference tasks.

---

## Multilingual BERT *(Devlin et al., 2018)*
- **Data**: Wikipedia in the top 104 languages.
- **Code-mixing**: Helps align words in different languages naturally.

---

## XLM *(Lample & Conneau, 2019)*
- **Objectives**: Masked LM + Translation LM.
- **Results**: Strong performance in cross-lingual classification and zero-shot scenarios.

---

## Concluding Remarks

| Model | Key Features & Applications | Repository / Links |
| :--- | :--- | :--- |
| **Transformer-XL** | Longer context dependency | [GitHub](https://github.com/kimiyoung/transformer-xl) |
| **XLNet** | AR + AE; No pretrain-finetune discrepancy | [GitHub](https://github.com/zihangdai/xlnet) |
| **RoBERTa** | Optimization details & data scaling | [GitHub](http://github.com/pytorch/fairseq) |
| **SpanBERT** | Better for QA, NLI, coreference | - |
| **Multilingual BERT**| Multi-language support | [GitHub](https://github.com/google-research/bert) |
| **XLM** | Zero-shot scenarios, Cross-lingual | [GitHub](https://github.com/facebookresearch/XLM) |