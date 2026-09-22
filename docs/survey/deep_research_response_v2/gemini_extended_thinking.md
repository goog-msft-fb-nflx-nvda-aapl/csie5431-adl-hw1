## **Few-Shot & Small-N Sentence-Embedding Fine-Tuning Strategies**

The English route currently operates under a severe data constraint with only 627 training examples. End-to-end fine-tuning using AutoModelForSequenceClassification attaches a randomly initialized linear head ($D \\to 10$) to the top of the backbone. Backpropagating classification loss through all layers on 627 examples tends to overfit or distort the underlying feature representations of strong sentence-embedding models like BAAI/bge-m3.

### **Embedding-Space Contrastive Pre-Tuning (SetFit / SupCon Framework)**

Instead of training a classification head directly from scratch on raw text, sentence-embedding classifiers benefit from a two-stage training paradigm:

> 1. **Supervised Contrastive Representation Fine-Tuning**: Fine-tune the BGE-M3 encoder using Supervised Contrastive Loss (SupCon) or multi-label cosine similarity loss. For pairs of English utterances $(x\_i, x\_j)$ with multi-label binary vectors $y\_i, y\_j \\in \\{0, 1\\}^{10}$, define target similarity based on Jaccard overlap:  
>    $$S\_{ij} \= \\frac{y\_i^T y\_j}{\\Vert{}y\_i\\Vert{}\_1 \+ \\Vert{}y\_j\\Vert{}\_1 \- y\_i^T y\_j}$$  
>    The model optimizes a cosine distance loss to pull utterances with overlapping intent combinations together while pushing non-overlapping pairs apart in embedding space.  
> 2. **Classifier Fitting on Frozen / Adapter-Tuned Embeddings**: Once the sentence embeddings are aligned to sales-intent similarity, extract the $1024$-dimensional pooled embeddings for all 627 training examples. Train 10 independent binary Logistic Regression or Ridge Classification heads (or a single calibrated linear layer) using scikit-learn.

### **Feature Extraction vs. Head Re-initialization**

* **Frozen Feature Extraction**: Extracting embeddings directly from BGE-M3 without fine-tuning encoder weights and training a scikit-learn LogisticRegression(C=1.0, class\_weight='balanced') or linear SVM provides a non-overfitting baseline.  
* **Pattern-Exploiting / Prompt-Based Classifier Heads**: Reformulating multi-label intent as cloze prompts (e.g., *"Customer intent: \[MASK\]. Utterance: {utterance}"*) works well for standard autoregressive or MLM encoders like RoBERTa or DeBERTa. However, for dual-encoder retrieval models like BGE-M3, contrastive pair tuning and feature extraction significantly outperform prompt-based MLM heads.

## **Fine-Tuning Instability on Small Datasets**

Fine-tuning transformer encoders on small datasets (under 1000 examples) suffers from high gradient variance, early representation collapse, and extreme sensitivity to initialization seeds and learning rates. This directly explains why the 627-example English route required a $5\\times$ higher learning rate ($1\\times 10^{-4}$ / $5\\times 10^{-5}$) than standard recommendations.

Layer-wise Learning Rate Decay (LLRD) Architecture  
\--------------------------------------------------  
\[ Top Classification Head \]   \--\> Learning Rate: η\_top \= η  
\[ Transformer Layer 12    \]   \--\> Learning Rate: η\_12  \= η · ξ^1  
\[ Transformer Layer 11    \]   \--\> Learning Rate: η\_11  \= η · ξ^2  
 ...  
\[ Transformer Layer 1     \]   \--\> Learning Rate: η\_1   \= η · ξ^12  
\[ Input Embeddings        \]   \--\> Learning Rate: η\_emb \= η · ξ^13

### **Layer-wise Learning Rate Decay (LLRD)**

LLRD applies a multiplicative decay factor $\\xi \\in (0.8, 0.9)$ to learning rates at deeper layers of the transformer stack. For layer $l \\in \\{1, 2, \\dots, L\\}$ (where $L$ is the top layer):

$$\\eta\_l \= \\eta\_0 \\cdot \\xi^{L \- l}$$  
This allows top-level classification weights to adapt aggressively to the 10 sales-intent targets while keeping lower-level multilingual syntactic and semantic features intact.

### **Top-Layer Re-initialization (Netzer et al. / Zhang et al.)**

The uppermost 1–2 layers of pretrained language models are heavily biased toward their original pretraining objectives (e.g., masked language modeling or dense retrieval contrastive loss). Re-initializing the top 2 transformer layers of the encoder with fresh Gaussian weights prior to fine-tuning on the 627 English examples removes task-agnostic head bias and prevents the optimization from getting trapped in poor local minima.

### **Optimization & Warmup Recipe for Small Datasets**

* **Extended Warmup**: Set linear warmup steps to 15%–20% of total training iterations to prevent large initial gradients from destroying pretrained weights.  
* **AdamW Weight Decay**: Set weight decay to $0.01$–$0.05$ with active bias correction to enforce regularization.  
* **Gradient Accumulation & Batch Size**: Use small physical batch sizes (8 or 16\) with gradient accumulation steps (2 or 4\) to achieve an effective batch size of 32 or 64, smoothing out gradient noise across rare labels.  
* **Macro-F1 Early Stopping**: Monitor validation Macro-F1 rather than loss. Empirical findings confirm training loss plateaus early while validation Macro-F1 continues to improve over 20–30 epochs.

## **Cross-Lingual Data Augmentation via Translate-Train**

Back-translation of the 627 English examples failed because paraphrasing existing inputs does not add new semantic variation or introduce rare label combinations. Translate-Train offers a far more effective alternative by leveraging the abundant \~1900 Chinese training examples.

                     Translate-Train Data Flow  
                     \-------------------------  
  \[ \~1900 Chinese Train Examples \] \----( Offline NMT )----\> \[ \~1900 Synthetic EN Examples \]  
                 |                                                        |  
                 v                                                        v  
  \[ Chinese Route Model \]                                  \[ Real EN (627) \+ Synthetic EN (\~1900) \]  
                                                                          |  
                                                                          v  
                                                                 \[ English Route Model \]

### **Translate-Train Strategy**

> 1. **Source Expansion**: Translate all \~1900 Chinese utterances from train\_split into English using an offline NMT model.  
> 2. **Dataset Combination**: Combine the \~1900 translated English utterances with the original 627 native English utterances, creating an augmented English training corpus of \~2500 examples.  
> 3. **Imbalance Correction for Rare Labels**: Chinese training data contains proportionally more instances of rare classes like Too\_Expensive and Compare\_Competitor. Translating these utterances directly triples the positive support for these bottleneck labels on the English route.

### **Managing Translation Artifacts & Label Transfer Fidelity**

* **Label Transfer Fidelity**: Intent labels represent high-level conversational goals (e.g., price inquiries or competitor comparisons). These semantics remain invariant under direct neural translation.  
* **Domain Loss Weighting**: Synthetic translations may contain "translationese" phrasing. Weight real native English examples at $1.0$ and synthetic translated examples at $0.5$ in the BCE/ASL loss calculation, or sample native English examples at a 2:1 ratio during batch creation:  
  $$L\_{total} \= L\_{native} \+ \\lambda L\_{translated}, \\quad \\lambda \\in \[0.5, 0.7\]$$  
* **Offline Execution**: Use Helsinki-NLP/opus-mt-zh-en or facebook/nllb-200-distilled-600M inside download.sh. Both models require $\<1.2$ GB disk space, install via HuggingFace transformers, and execute offline in seconds for 1900 short text strings.

## **Rare-Class Performance in Multi-Label Settings**

The two weakest classes on the English route—Too\_Expensive and Compare\_Competitor—suffer from extreme class imbalance (\~6–7% prevalence). Beyond standard weighted BCE and Asymmetric Loss (ASL), multi-label loss formulations must account for both positive/negative imbalance and hard negative gradients.

                         Multi-Label Loss Comparison  
\----------------------------------------------------------------------------------  
 Loss Function            Primary Mechanism               Best Suited For  
\----------------------------------------------------------------------------------  
 Plain BCE                Uniform element-wise loss       Balanced multi-label data  
 Asymmetric Loss (ASL)    Downweights easy negatives      Moderate imbalance (English route)  
 Distribution-Balanced    Frequency & margin adjustment   Extreme imbalance & rare labels  
 Class-Balanced (Cui)     Effective sample count weighting Class prevalence skew  
\----------------------------------------------------------------------------------

### **Distribution-Balanced Loss (DB-Loss, Wu et al.)**

Distribution-Balanced Loss modifies standard binary cross-entropy by addressing both class-frequency imbalance and negative-to-positive ratio imbalance. For class $k \\in \\{1, \\dots, 10\\}$ with logit $z\_k$:

$$L\_{DB} \= \-\\sum\_{k=1}^{10} \\left\[ y\_k \\log \\hat{P}\_k \+ (1 \- y\_k) \\log (1 \- \\hat{P}\_k) \\right\]$$  
where the re-balanced probability $\\hat{P}\_k$ is computed using a class-specific soft margin $v\_k$ and gradient re-weighting factor $r\_k$:

$$\\hat{P}\_k \= \\frac{1}{1 \+ e^{-(z\_k \- v\_k)}}, \\quad v\_k \= \\log \\left( \\frac{N \- N\_k}{N\_k} \\right)^\\alpha$$  
Here $N\_k$ is the positive sample count for class $k$, $N$ is total training examples, and $\\alpha \\in \[0.1, 0.5\]$ controls margin sharpness. This margin shift lowers the decision boundary for Too\_Expensive and Compare\_Competitor, preventing dominant negative instances from suppressing rare label activations.

### **Prototype-Based Class Heads**

For rare classes with fewer than 40 positive instances on the English route, standard linear projection heads struggle to establish stable decision hyperplanes. A nearest-prototype head establishes class centroids directly in embedding space:

> 1. Calculate class prototype vectors $p\_k$ for each label $k \\in \\{1, \\dots, 10\\}$ using BGE-M3 representations of positive training instances $S\_k$:  
>    $$p\_k \= \\frac{1}{\\vert{}S\_k\\vert{}} \\sum\_{i \\in S\_k} \\frac{f(x\_i)}{\\Vert{}f(x\_i)\\Vert{}\_2}$$  
> 2. For an input utterance embedding $f(x)$, compute probability of label $k$ as a temperature-scaled cosine similarity:  
>    $$\\hat{y}\_k \= \\sigma \\left( \\tau \\cdot \\frac{f(x)^T p\_k}{\\Vert{}f(x)\\Vert{}\_2} \+ b\_k \\right)$$  
> 3. Fine-tune scale parameter $\\tau$ and bias terms $b\_k$ on validation split dev.

## **Literature Benchmarks & SOTA Dialogue Intent Paradigms**

Multi-label intent classification on short dialogue utterances is widely studied across intent identification benchmarks such as MASSIVE, DialoGLUE, HWU64, and MultiWOZ. Key architectural insights from high-performing competition entries and papers include:

### **Label-Aware Attention Networks (LAAN) / Zero-Shot Semantic Anchoring**

Standard classifiers project text representations into an abstract label vector space via an unconstrained weight matrix $W \\in \\mathbb{R}^{D \\times 10}$. Label-Aware Attention replaces this matrix with text embeddings of the label descriptions themselves:

                  Label-Aware Attention Architecture  
                  \----------------------------------  
\[ Utterance: "Is this cheaper than Brand X?" \] \---\> \[ Encoder f(x) \] \---\> Vector u (D-dim)  
                                                                               |  
\[ Label Text: "Compare with competitor"     \] \---\> \[ Encoder f(l) \] \---\> Vector l\_k (D-dim)  
                                                                               |  
                                                 Cosine Similarity: s\_k \= cos(u, l\_k)  
                                                                               |  
                                                 Sigmoid Activation: y\_k \= σ(a · s\_k \+ b)

By initializing label weights using embedding representations of human-readable label names (e.g., *"Compare with competitor"* or *"Product is too expensive"*), the model establishes a zero-shot semantic anchor. The linear layer only needs to adjust scalar scaling factors rather than learning dense $1024$-dimensional feature projections from scratch.

### **Label Co-occurrence & Classifier Chains**

In sales dialogues, specific intent pairs frequently co-occur (e.g., Ask\_Price co-occurring with Too\_Expensive or Ask\_Spec co-occurring with Compare\_Competitor). Standard independent binary classifiers ignore these correlations.

* **Classifier Chains**: Order labels by frequency and pass predictions of dominant labels as auxiliary features to downstream rare-label classifiers.  
* **Interaction Matrix Regularization**: Penalize prediction outputs that violate empirical label co-occurrence probability matrices derived from train\_split.

## **HuggingFace Model Search & Resource Budgeting**

All models must fit within strict execution and hardware constraints: Ubuntu 20.04, RTX 3080 Ti (10GB VRAM), max 4GB total download during download.sh, PyTorch 2.11.0, and transformers 4.50.0.

| Model Identifier | Architecture / Type | Parameters | Precision | VRAM (Batch 16\) | Disk Footprint | Key Advantage |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **BAAI/bge-m3** | Dense \+ Multi-vector Retrieval | 568M | fp16 | \~3.2 GB | \~1.14 GB | **Current Best**; top retrieval & embedding backbone. |
| **intfloat/multilingual-e5-large** | Dense Multilingual Encoder | 560M | fp16 | \~3.1 GB | \~1.12 GB | Strong cross-lingual alignment; handles query: prefixes. |
| **Alibaba-NLP/gte-multilingual-base** | Long-context Encoder | 305M | fp16 | \~1.8 GB | \~0.61 GB | Lightweight; 8k context; leaves download budget for NMT. |
| **sentence-transformers/LaBSE** | Dual-Encoder Cross-Lingual | 471M | fp16 | \~2.6 GB | \~0.94 GB | Specifically optimized for cross-lingual vector alignment. |
| **intfloat/multilingual-e5-base** | Dense Multilingual Encoder | 278M | fp16 | \~1.6 GB | \~0.56 GB | Fast iteration speed (30s per epoch on RTX 3080 Ti). |

### **Download & VRAM Budget Allocation (download.sh)**

To remain strictly within the 4GB download limit and 10GB VRAM ceiling:

* Primary English & Chinese Classifier (BAAI/bge-m3 or gte-multilingual-base): \~1.14 GB.  
* Translation Model (Helsinki-NLP/opus-mt-zh-en): \~300 MB.  
* Secondary Backup Classifier (intfloat/multilingual-e5-base): \~0.56 GB.  
* **Total Download Size**: \~2.00 GB (well below the 4GB cap).

## **Prioritized Action Plan**

Given the RTX 3080 Ti GPU hardware environment (1–5 minutes per training run), the following step-by-step roadmap prioritizes maximum Macro-F1 improvement for minimum engineering effort:

> 1. **Implement Translate-Train Augmentation (Highest Expected Impact)**  
   * Use Helsinki-NLP/opus-mt-zh-en in download.sh to translate all \~1900 Chinese training examples into English offline.  
   * Train the English route on the combined \~2500 dataset (627 native \+ \~1900 synthetic).  
   * Apply domain loss weighting ($\\lambda \= 0.6$) or a 2:1 batch sampling ratio in favor of native English examples.  
> 2. **Apply Distribution-Balanced Loss (DB-Loss) on English Route**  
   * Replace Asymmetric Loss (ASL) on the English route with Distribution-Balanced Loss.  
   * Tune margin scale $\\alpha \\in \[0.1, 0.3\]$ to boost sensitivity on Too\_Expensive and Compare\_Competitor.  
> 3. **Implement Layer-wise Learning Rate Decay (LLRD) & Top-Layer Re-initialization**  
   * Apply LLRD with decay $\\xi \= 0.85$ and top-layer re-initialization (re-initializing top 2 layers of BGE-M3).  
   * Set peak learning rate to $5\\times 10^{-5}$ with a 15% linear warmup schedule.  
> 4. **Evaluate Alibaba-NLP/gte-multilingual-base as Alternative Backbone**  
   * Benchmark gte-multilingual-base against BAAI/bge-m3 on the expanded Translate-Train dataset.  
   * Ensembling BGE-M3 predictions with gte-multilingual-base via logit averaging yields strong model diversity on short dialogue text\[cite: 1\].