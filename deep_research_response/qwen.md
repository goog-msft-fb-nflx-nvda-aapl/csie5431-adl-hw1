Here is a comprehensive, constraint-aware research brief addressing your six questions, with citations to papers, Hugging Face model cards, and dataset repositories.

---

### 1. EDA — What You Might Be Missing
Your initial EDA is strong, but a strong practitioner would add the following before modeling:
*   **Tokenizer-Based Length Distributions**: Character counts are highly misleading for bilingual text. Chinese characters often map to 1–3 tokens in models like XLM-R, while English subwords vary. You must tokenize the entire dataset with your *target* tokenizer (e.g., `xlm-roberta-base`) to find the true 90th/99th percentile token lengths. If the 99th percentile exceeds 512 tokens, your truncation strategy will actively discard data.
*   **Per-Class Length/Context Correlation**: Do not assume all intents behave the same. `Need_More_Evidence` or `Compare_Competitor` may have significantly longer prior contexts than `Ask_Price`. Plot mean token length per label to see if truncation will disproportionately harm specific classes.
*   **Keyword/N-gram Analysis for Confusable Pairs**: Run TF-IDF or simple n-gram extraction for pairs like `Doubt` vs. `Need_More_Evidence`. Look for linguistic markers (e.g., Chinese "真的吗" vs. "有没有具体数据"; English "Are you sure?" vs. "Can you show me proof?"). This informs whether the model needs to attend to specific lexical triggers.
*   **Label Co-occurrence as a Correlation Matrix**: Plot a 10x10 heatmap of label co-occurrence (Jaccard similarity or pointwise mutual information). This reveals if certain pairs are mutually exclusive or highly correlated, which can inform loss functions (e.g., asymmetric loss) or post-processing rules.
*   **Red Flags for Distribution Shift**: The assignment’s warning about "different distribution" likely extends beyond the 75/25 vs. 49/51 language ratio. You must explicitly check:
    1.  **Domain shift**: Does the test set contain more financial/insurance dialogues compared to the train set’s e-commerce focus? (Check for domain-specific keywords).
    2.  **Cardinality shift**: Does the average number of labels per utterance change between train and test?
    3.  **Context emptiness shift**: You noted ~10% empty context in train; verify this holds exactly in the public test.

---

### 2. SOTA / Practical Model Choices (Under 10GB VRAM / 4GB Download)
Given your strict compute and download constraints, encoder-based models are the most credible choice. Small decoder LLMs are risky for strict multi-label classification under a 2-hour inference budget.

| Model | HF Repo ID | Params | Disk Size (fp32) | License | Notes on Intent/Multi-label Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XLM-RoBERTa-base** | `xlm-roberta-base` | ~279M | ~1.12 GB | Apache 2.0 | The gold standard for bilingual (zh/en) tasks. Pretrained on 2.5TB of CommonCrawl across 100 languages. Proven strong on multilingual intent and dialogue-act classification. [[7]], [[8]] |
| **mBERT** | `bert-base-multilingual-cased` | ~178M | ~714 MB | Apache 2.0 | Smaller and faster than XLM-R, but slightly weaker on zero-shot cross-lingual transfer. A safe, highly efficient baseline. [[10]] |
| **BGE-M3** | `BAAI/bge-m3` | ~570M | ~2.27 GB | MIT / Apache 2.0 | State-of-the-art multilingual, multi-granularity embedding model. Can be fine-tuned for classification, but its larger size may require gradient checkpointing to fit 10GB VRAM with large batches. [[91]], [[94]] |
| **Chinese RoBERTa** | `hfl/chinese-roberta-wwm-ext` | ~100M+ | ~400 MB | Apache 2.0 | Excellent for Chinese-only ablations. Uses whole-word masking, outperforming standard `bert-base-chinese` on Chinese NLU tasks. [[29]], [[34]] |
| **DeBERTa-v3-base** | `microsoft/deberta-v3-base` | ~184M | ~700 MB | MIT | SOTA for English intent/text classification due to improved disentangled attention. Weak on Chinese unless extended, but great for English ablation. [[38]], [[44]] |

**Decoder LLMs (Qwen2.5-0.5B/1.5B, Gemma-2-2B)**: While `Qwen2.5-0.5B` (~500MB) or `Gemma-2-2B` (~2.6B params, ~1.6GB quantized) are viable for few-shot prompting, literature shows that for *strict, high-throughput multi-label classification*, fine-tuned BERT-style encoders consistently match or exceed small LLMs in accuracy while being 5–10x faster at inference and far more stable in output formatting. [[56]], [[62]] Given the 2-hour inference limit and Macro-F1 grading gate, a fine-tuned encoder is the safer, more practical choice.

---

### 3. Public Datasets for Auxiliary Use
*Constraint Check*: You cannot use a model *already trained* on a sales-intent dataset, but you can use public datasets for auxiliary pretraining or to inform augmentation, provided you train the final classifier on your target data.

*   **JDDC (JD Dialogue Dataset)**: A large-scale, real-world Chinese e-commerce conversation corpus with over 1 million multi-turn dialogues, including dialogue act (intent) annotations. [[96]], [[98]]. *License*: Requires registration/agreement for research use via JD AI. *Usefulness*: Highly relevant for Chinese e-commerce intent; excellent for domain-adaptive pretraining (DAPT) of your encoder before fine-tuning on your 3000 examples.
*   **ECD (E-commerce Dialogue Corpus)**: Extracted from real Taobao customer-service conversations, preserving natural conversation flow and intent structures. [[104]], [[105]]. *License*: Research use via official academic release. *Usefulness*: Similar to JDDC, great for Chinese dialogue act regularization.
*   **MultiWOZ (2.1/2.2)**: A fully-labeled collection of human-human written conversations spanning multiple domains, with dialogue act annotations (e.g., `inform`, `request`). [[65]]. *License*: Apache 2.0. *Usefulness*: Good for general English dialogue act pretraining, though the schema differs from your 10 labels.
*   **Schema-Guided Dialogue (SGD)**: 18k multi-domain task-oriented dialogues with intent and slot annotations. [[64]]. *License*: Apache 2.0. *Usefulness*: Useful for English intent augmentation.

*Recommendation*: Given the 4GB download limit and 1-hour fetch time, downloading massive datasets like JDDC might blow your budget. Instead, consider using these datasets *only* if you can find a pre-filtered, smaller subset on Hugging Face, or use them to train a lightweight, separate T5-style paraphrase model for data augmentation (if compute allows).

---

### 4. Long-Input Handling Strategies
With a 512-token window and a tail up to 7220 characters, naive truncation will fail. Standard practical strategies include:
1.  **Head + Tail Truncation**: Keep the first $N$ tokens (to preserve initial context/system prompts) and the last $M$ tokens (to preserve the current utterance and immediate prior turns). This is empirically superior to tail-only truncation for text classification. [[72]]
2.  **Turn-Level Truncation (Keep-Last-K)**: Instead of truncating by tokens, parse the context list and keep only the last 2 or 3 turns + the current utterance. This is semantically meaningful and guarantees the current utterance is never truncated.
3.  **Hierarchical Encoding**: Encode each turn independently using a shared encoder (e.g., XLM-R), then pool the turn representations (mean/max) or pass them through a lightweight GRU/Transformer to get a dialogue-level representation. This completely bypasses the 512-token limit and is highly VRAM-efficient.
4.  **Which intents need long context?** Literature and practice suggest `Compare_Competitor`, `Need_More_Evidence`, and `Doubt` are highly context-dependent (e.g., "But you just said..."). `Ask_Price`, `State_Need`, and `Confirm_Order` are often decidable from the current utterance alone. Tailor your truncation to preserve recent turns for the former.

---

### 5. Imbalanced Multi-Label Training & Macro-F1 Optimization
Your label distribution (6.6% to 34.9%) combined with a strict Macro-F1 gate requires specific techniques:
*   **Loss Function**: Use **Asymmetric Loss (ASL)** or **Focal Loss** instead of standard BCE. ASL dynamically down-weights easy negatives and focuses on hard positives/negatives, explicitly designed to handle the high negative-positive imbalance in multi-label settings. [[79]], [[82]]. If ASL is too complex to implement from scratch, `BCEWithLogitsLoss` with `pos_weight` calculated inversely proportional to class frequency is a strong, simpler alternative.
*   **Per-Class Threshold Tuning**: **Do not use a flat 0.5 threshold.** Rare classes will have lower predicted probabilities. After training, use a held-out validation split (e.g., 10% of train) to search for the optimal decision threshold *per class* that maximizes Macro-F1. [[80]], [[85]]
*   **Sampling**: Avoid MLSMOTE or heavy oversampling on a small 3000-example dataset, as it极易 leads to severe overfitting. Class-weighted loss is more robust here.
*   **Macro vs. Micro Trade-off**: Macro-F1 is dragged down by rare class failures; Micro-F1 is dominated by frequent classes. ASL + per-class threshold tuning is the most proven way to satisfy *both* simultaneously, as it forces the model to care about the rare classes without destroying overall accuracy.

---

### 6. Robustness to Train/Test Language-Ratio Shift
The shift from 75% zh / 25% en (train) to ~49% zh / 51% en (test) is a classic covariate shift.
*   **Language-Balanced Batch Sampling**: During fine-tuning, do not sample randomly. Implement a custom sampler or stratified batching to ensure every batch (or every epoch) has a ~50/50 zh/en ratio. This prevents the optimizer from overfitting to Chinese gradients and under-updating on English patterns. [[117]]
*   **Encoder Choice**: A genuinely multilingual encoder (`xlm-roberta-base` or `BGE-M3`) is strictly required. Do not attempt to ensemble monolingual models, as they do not share a semantic space and will fail on code-switched or ambiguous inputs.
*   **Alternative Interpretation of "Different Distribution"**: The assignment hint might imply more than just language ratio. You must verify:
    1.  **Domain Shift**: Are there new topics in the test set (e.g., insurance vs. e-commerce)?
    2.  **Context Length Shift**: Is the test set systematically longer or shorter?
    3.  **Label Co-occurrence Shift**: Does the test set feature different combinations of labels (e.g., `Too_Expensive` + `Doubt` is more common in test)?

---

### Prioritized Recommendation (What to Try First)
Given your constraints (10GB VRAM, 4GB download, 2-hour inference, strict Macro/Micro F1 gates):
1.  **Model**: Start with **`xlm-roberta-base`**. It is ~1.12 GB (well under the 4GB limit), natively handles zh/en, and fits comfortably in 10GB VRAM with a batch size of 16–32 using gradient checkpointing.
2.  **Preprocessing**: Implement **Keep-Last-3-Turns + Current Utterance** truncation. If the total exceeds 512 tokens, apply **Head+Tail truncation** (e.g., keep first 64 tokens of the oldest retained turn, and the last 448 tokens). This guarantees the current utterance is never lost.
3.  **Training**: Use **`BCEWithLogitsLoss` with class weights** (inverse frequency). Train for 3–5 epochs with a low learning rate (e.g., 2e-5) and early stopping.
4.  **Post-Processing**: Dedicate 10% of your training data as a validation set. After training, run a grid search over thresholds `[0.1, 0.2, ..., 0.9]` for *each of the 10 classes* independently to find the combination that maximizes **Macro-F1** while keeping Micro-F1 above the threshold.
5.  **Robustness**: Use a simple stratified sampler to ensure each training batch contains a balanced mix of Chinese and English examples.

### Uncertainties to Double-Check Yourself
*   **PyTorch Version**: You listed `PyTorch 2.11.0`. This version does not exist (current stable is 2.x, likely 2.1.x or 2.2.x). Verify if this is a typo in the assignment brief. If it is a strict string check, it may be a trick question or a typo for `2.1.0`.
*   **Exact Disk Footprint**: Before finalizing `download.sh`, run a quick script to download your chosen model and tokenizer and check the exact directory size (e.g., `du -sh ~/.cache/huggingface/hub/...`) to guarantee it is strictly under 4GB.
*   **JDDC/ECD Licensing**: If you decide to use JDDC or ECD, double-check their specific academic license terms to ensure your usage (even for auxiliary pretraining) complies with their redistribution rules, or stick strictly to the provided 3000 examples to avoid any rule violations.