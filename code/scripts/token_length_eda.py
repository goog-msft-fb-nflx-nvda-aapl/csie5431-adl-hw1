import json

import numpy as np
from transformers import AutoTokenizer

from data_utils import build_context_text, load_jsonl

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "macbert": "hfl/chinese-macbert-base",
    "roberta": "roberta-base",
}


def percentiles(arr):
    arr = np.array(arr)
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main():
    examples = load_jsonl("/home/jtan/adl_hw1/data/train.jsonl")
    results = {}
    for key, hf_id in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf_id)
        full_lens, utt_lens, zh_full, en_full = [], [], [], []
        over512 = 0
        for e in examples:
            ctx = build_context_text(e["context"], k=None)
            utt = f"[USER] {e['utterance']}"
            full_text = f"{ctx}\n{utt}" if ctx else utt
            full_ids = tok.encode(full_text)
            utt_ids = tok.encode(utt)
            full_lens.append(len(full_ids))
            utt_lens.append(len(utt_ids))
            if len(full_ids) > 512:
                over512 += 1
            (zh_full if e["language"] == "zh" else en_full).append(len(full_ids))
        results[key] = {
            "full_context_utt": percentiles(full_lens),
            "utterance_only": percentiles(utt_lens),
            "pct_over_512": 100 * over512 / len(examples),
            "full_zh": percentiles(zh_full),
            "full_en": percentiles(en_full),
        }
        print(f"=== {key} ({hf_id}) ===")
        print(json.dumps(results[key], indent=2))

    with open("/home/jtan/adl_hw1/results/token_length_eda.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
