# User State Prediction
**Curriculum Number:** CSIE5431 
**Item:** A1

## Env
 - GPU server can be accessed via ssh gsm-gpu2
 - there is no sudo permission on the GPU
 - use tmux and conda to run experiment and environment creation.
 - only modify under jtan, do not modify other user or shared data.
 jtan@TO-sv-td-h200nvlnode02:~$ pwd
/home/jtan
 - if you need to use GitHub or HuggingFace access token to download model/code repo, let me know in chat, i will provide.

## Assignments & Final Project
- **Goal:** multi-agent game
  - **Sales agent**
    - A1: Transformer/BERT for user state prediction
    - A2: RAG for product knowledge access
  - **Customer agent**
    - A3: LM fine-tuning for tool-calling

## Task – Sales-Conversation Intent Classification
- **Labels**
  - 10 labels: `Ask_Price`, `Ask_Service`, `Ask_Spec`, `Compare_Competitor`, `Confirm_Order`, `Doubt`, `Need_More_Evidence`, `Need_Time_To_Think`, `State_Need`, `Too_Expensive`
  - One utterance may have 0~K labels (K > 1).
- **Class Distribution**
  - The data is naturally imbalanced.
  - Train and test sets may have different distributions.
- **Context**
  - Some samples include dialogue context, others do not.
  - The data may contain realistic noise and ambiguity.

## Data Examples

| Utterance | Label | Why |
| :--- | :--- | :--- |
| “Is this really waterproof?” | Doubt | Questions the claim |
| “What is the waterproof rating?” | Ask_Spec | Asks for an attribute |
| “Do you have a waterproof test report?” | Need_More_Evidence | Requests supporting evidence |
| “How much is it, and does it come with a warranty?” | Ask_Price, Ask_Service | Ask the price and about after-sales support |
| 「另一家賣得比你們便宜 200塊，你們這個有點貴」 | Compare_Competitor, Too_Expensive | Compares with another shop and says this one is pricey |
| 「好，那我下個月再來看看。」 | Need_Time_To_Think | Puts off the decision until next month |

## Data & Evaluation Metrics
- Data: ntu-adl-2026-hw1-1.zip
- **Evaluation metrics:** Macro-F1 / Micro-F1
  - Please make sure you know how the metrics are computed.

## Report Questions
- **Model variants (1%)**
  - Compare different models, inputs, or training variants.
  - Discuss the observed differences.
- **Does dialogue context help (1%)**
  - Compare with and without dialogue context.
  - Analyze which classes benefit most.
- **Long inputs (1%)**
  - Compare different strategies for handling long inputs.
  - Report performance and affected classes.
- **Chinese vs. English (1%)**
  - Compare performance on Chinese/English data using different pretrained models.
- **Additional Exploration (2%)**
  - Explore any relevant direction beyond the above.
  - Focus on findings & insights.

## Rules - What You Can Do
- You may use external datasets or resources, but SalesLLM[https://github.com/Bairong-Xdynamics/Benchmarking-LLM-Realistic-Selling-Skill] is not allowed. Clearly describe any external data used in your report.
- Use publicly available pre-trained LMs.
- **Allowed packages:**
  - Python 3.12 and Python Standard Library
  - PyTorch 2.11.0, scikit-learn 1.9.0
  - tqdm, numpy, pandas
  - transformers 4.50.0, datasets 2.21.0, accelerate 0.34.2
  - evaluate, matplotlib, gdown
  - Any dependencies required by the packages listed above

## Rules - What You Cannot Do
- Any means of cheating or plagiarism.
- Use the labels of the test data directly or indirectly. *(Do not try to find them.)*
- Search for, retrieve, or manually annotate data from the source dataset to obtain additional labeled data.
- Use datasets, packages, or tools not allowed.
- Use models trained with other intent/sales datasets, including but not limited to:
  - MultiSense/SaleIntent_bert: https://huggingface.co/MultiSense/SaleIntent_bert
  - *If not sure, ask TA first.*
- Give/get trained model/predictions to/from others.
- Publish your code before deadline.
- *Violations may cause zero/negative score and punishment from school.*

## Submission
Zip your folder, which should be named as your student id (lower-cased) (ex. `r14000000`) and submit the `.zip` to NTU Cool. Your folder should contain:
- `README.md`
- `run.sh`
- `download.sh`
- `report.pdf`
- Your code/script (all the code/script you used to train, predict, or plot report figures should be included).
- **Do not upload training, testing data or models to COOL.**

### Submission - download.sh
- `download.sh` should download your models, tokenizers and data.
- You can upload your models/tokenizers/data to:
  - Dropbox and use `wget` to download.
  - Google Drive and use `gdown` to download.
- Please make sure we have the access to download!
- Do not modify the files in download links after the deadline, this action is considered cheating.
- Keep the download links in `download.sh` valid for at least 2 weeks after deadline.
- Do not do things more than downloading, otherwise, your `download.sh` may be killed.
- You can download at most 4GB, and `download.sh` should finish within 1 hour. *(At csie dept. with maximum 10MB/s bandwidth)*
- We will execute `download.sh` before running any other scripts.

### Submission - run.sh
- `run.sh` performs inference using your trained models and output predictions in `test.json`.
- **3 required arguments:**
  - `"${1}"`: path to `context.json`.
  - `"${2}"`: path to `test.json`.
  - `"${3}"`: path to the output prediction file named `prediction.csv`.
- **TA will predict testing data as follow:**
  ```bash
  bash ./download.sh
  bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
  ```
- `run.sh` should finish within 2 hours. *(See Execution Environment for details)*
- Make sure your code works!

### Submission - Execution Environment
- Your code/script will be run on a server with:
  - Ubuntu 20.04
  - 32GB RAM, RTX 3080 Ti 10GB VRAM, and 20GB disk space available.
- The packages we allow only.
- Python 3.10
- No network access after we run `download.sh`.

### Submission - README.md
- `README.md` should contain step-by-step instruction on how to train your model with your codes/scripts.
- You will get a **-2 penalty** if you have no or empty `README.md`.
- If necessary, you will be required to reproduce your results based on the `README.md`.
- If you cannot reproduce your result, you may lose points.

## Grading

### Model Performance (14%)
- Public test set (7%); Private test set (7%)
- Performance is graded by difficulty levels based on Macro-F1 and Micro-F1. All thresholds must be satisfied simultaneously.
- TA can reproduce your results without human intervention. **0 point** if we cannot reproduce your submission after human intervention.

| Level | Macro-F1 | Micro-F1 |
| :--- | :--- | :--- |
| **Basic (3%)** | ≥ 0.62 | ≥ 0.73 |
| **Medium (3%)** | ≥ 0.76 | ≥ 0.79 |
| **Challenge (1%)** | ≥ 0.82 | ≥ 0.83 |

### Report (6%)
- Discussion and analysis for report questions
  - Please provide concrete evidences for your answers
- Submit in PDF format. [Use Overleaf Latex to Generate the PDF]


Submission: 
<student-id>.zip
└── <student-id>/
    ├── README.md
    ├── download.sh
    ├── run.sh
    ├── report.pdf
    └── code/scripts

 

Important Notes:
This is a multi-label intent classification task. 
Do not use test labels or prohibited datasets/models.
Make sure download.sh & run.sh work currently. 
Please refer to HW1 slides / README for detailed instructions. 