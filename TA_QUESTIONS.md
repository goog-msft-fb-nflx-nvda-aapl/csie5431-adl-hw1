# Draft questions for TA — ADL HW1 (A1)

Drafted from gaps/ambiguities surfaced during EDA + the Deep Research survey (see `WORKLOG.md`). Not yet sent.

1. **Is `public_test_gold.csv` usable for model selection / threshold tuning, or diagnostic-only?**
   You give us the public test's ground-truth labels directly. The rules say "do not use the labels of the test data directly or indirectly," which reads as it should cover the private test set — but it's ambiguous whether that also restricts how we use the *public* test gold labels you've already released. Specifically: is it acceptable to tune per-class decision thresholds or do model/checkpoint selection against `public_test_gold.csv`, or should we treat it as look-but-don't-touch (diagnostic only) and do all tuning on a held-out split carved from `train.jsonl`? This changes our pipeline design, so we'd like to know before starting training.

2. **Does "use publicly available pre-trained LMs" extend to auxiliary/unsupervised use of *other public datasets*, not just models?**
   We're considering two non-label-transfer uses of external dialogue datasets (e.g. MultiWOZ, SGD, or Chinese e-commerce corpora) that are not sales-intent datasets and not the banned SalesLLM benchmark:
   - (a) continued/domain-adaptive MLM pretraining on their raw dialogue text (no labels used at all), and
   - (b) auxiliary supervised pretraining of the encoder body on their own dialogue-act/intent labels (a different label schema than ours), then discarding that head before fine-tuning on our 10 labels.
   Is either of these in bounds under "you may use external datasets or resources" (with a "clearly describe any external data used" requirement), or does the "cannot use a model trained with other intent/sales-intent datasets" rule extend to this kind of auxiliary pretraining as well?

3. **Is train-data-derived paraphrase/back-translation augmentation allowed?**
   Given the rule against "search for, retrieve, or manually annotate data from the source dataset," we want to confirm that *generating new synthetic utterances from our own labeled `train.jsonl` rows* (e.g. paraphrasing or back-translating an existing utterance, keeping its original label) is fine, as distinct from retrieving new labeled examples from wherever the original data came from. Any constraints on which models we can use to generate these paraphrases (open pretrained LM vs. proprietary API)?

4. **Any restriction on gated/click-through-license pretrained models?**
   Some multilingual/decoder candidates we're considering (e.g. Google Gemma, Meta Llama 3.2) require accepting a click-through license on Hugging Face before download. Since our own `download.sh` would fetch the already-downloaded weights from our Dropbox/Google Drive (not from the gated source at grading time), is there any issue using such a model, or should we stick to fully open-license models (Apache-2.0/MIT) to be safe?

5. ~~**Minor: `PyTorch 2.11.0` in the allowed-packages list — please confirm this is the intended version.**~~
   Checked ourselves — `torch==2.11.0` is a real released version (PyPI has it), no issue. Withdrawn.

8. **`scikit-learn==1.9.0` requires Python ≥3.11, but the grading env is specified as Python 3.10.**
   Verified against PyPI directly (not a guess): `scikit-learn` 1.9.0's wheels declare `Requires-Python >=3.11`, so `pip install scikit-learn==1.9.0` fails outright under Python 3.10. Since our own `run.sh` won't need to `pip install` anything at grading time (packages are presumably pre-installed in your environment), this may be moot for us — but wanted to flag the version mismatch in case it's a copy-paste artifact in the spec, or in case you intend for us to `pip install` these ourselves. For our dev environment we're using `scikit-learn==1.7.2` (latest compatible with Python 3.10) as a substitute; happy to switch if you confirm the intended pin.

6. **"Chinese vs. English" report question — scope check.**
   Does this expect (a) per-language evaluation of a single bilingual model, (b) a comparison against separately-trained monolingual models (e.g. Chinese MacBERT vs. English RoBERTa), or (c) both? We plan to report both unless told otherwise, just confirming that satisfies the intent of the question.

7. **"Model variants" report question — scope relative to the lecture (`Bert-Variant.md`)?**
   The BERT Variants lecture covers Transformer-XL, XLNet, RoBERTa, SpanBERT, Multilingual BERT, and XLM. Newer/stronger models we're otherwise considering (e.g. XLM-R, mDeBERTa-v3, BGE-M3) aren't in the slides. Is the "model variants" comparison meant to stay within the taught architectures, or is reaching beyond them (while still grounding the discussion in the lecture's concepts) expected/rewarded?
