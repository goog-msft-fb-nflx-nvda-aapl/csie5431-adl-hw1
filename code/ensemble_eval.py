import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import filter_lang, load_json, load_jsonl
from eval_metrics import compute_f1, sigmoid
from models import LABELS, MODEL_REGISTRY
from predict_final import InferDataset, collate
from threshold_tune import tune_thresholds
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def run_forward(run_name, examples, device, labeled=True):
    with open(f"{RESULTS_ROOT}/{run_name}/config.json") as f:
        config = json.load(f)
    hf_id = MODEL_REGISTRY[config["model_key"]]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(LABELS), problem_type="multi_label_classification"
    ).to(device)
    model.load_state_dict(torch.load(f"{RESULTS_ROOT}/{run_name}/best_model.pt", map_location=device))
    model.eval()

    ds = InferDataset(examples, tokenizer, config["context_mode"], config["k"], config["max_length"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--dev_path", default="/home/jtan/adl_hw1/data/dev_split.jsonl")
    ap.add_argument("--context_path", default="/home/jtan/adl_hw1/data/context.json")
    ap.add_argument("--test_path", default="/home/jtan/adl_hw1/data/test.json")
    ap.add_argument("--gold_path", default="/home/jtan/adl_hw1/data/public_test_gold.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev_examples = filter_lang(load_jsonl(args.dev_path), args.lang)
    dev_labels = np.array([[1.0 if l in e["labels"] else 0.0 for l in LABELS] for e in dev_examples])

    dev_probs_list = []
    for run in args.runs:
        logits = run_forward(run, dev_examples, device)
        dev_probs_list.append(sigmoid(logits))
    avg_dev_probs = np.mean(dev_probs_list, axis=0)

    thresholds, base_metrics, micro_floor = tune_thresholds(avg_dev_probs, dev_labels)
    tuned_preds = (avg_dev_probs >= thresholds[None, :]).astype(int)
    tuned_metrics = compute_f1(dev_labels, tuned_preds)
    print(f"ensemble {args.runs} lang={args.lang}")
    print(f"dev flat-0.5   macro={base_metrics['macro_f1']:.4f} micro={base_metrics['micro_f1']:.4f}")
    print(f"dev tuned      macro={tuned_metrics['macro_f1']:.4f} micro={tuned_metrics['micro_f1']:.4f}")

    context = load_json(args.context_path)
    test = json.load(open(args.test_path))
    test_examples = [e for e in test if e["language"] == args.lang]
    for e in test_examples:
        e["context"] = context.get(e["id"], [])

    test_probs_list = []
    for run in args.runs:
        logits = run_forward(run, test_examples, device)
        test_probs_list.append(sigmoid(logits))
    avg_test_probs = np.mean(test_probs_list, axis=0)
    test_preds = (avg_test_probs >= thresholds[None, :]).astype(int)

    import csv
    gold = {}
    with open(args.gold_path) as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = [int(row[l]) for l in LABELS]
    gold_arr = np.array([gold[e["id"]] for e in test_examples])
    test_metrics = compute_f1(gold_arr, test_preds)
    print(f"public-test {args.lang}-subset tuned macro={test_metrics['macro_f1']:.4f} micro={test_metrics['micro_f1']:.4f}")

    np.save(f"/tmp/ensemble_{args.lang}_dev_probs.npy", avg_dev_probs)
    with open(f"/tmp/ensemble_{args.lang}_thresholds.json", "w") as f:
        json.dump({LABELS[i]: float(thresholds[i]) for i in range(len(LABELS))}, f, indent=2)


if __name__ == "__main__":
    main()
