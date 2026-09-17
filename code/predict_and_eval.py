import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import labels_to_vector, load_json, load_jsonl
from eval_metrics import compute_f1, sigmoid
from models import LABELS, MODEL_REGISTRY
from train import IntentDataset, collate

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def build_test_examples(context_path, test_path):
    context = load_json(context_path)
    test = load_json(test_path)
    out = []
    for e in test:
        out.append({
            "id": e["id"], "language": e["language"], "utterance": e["utterance"],
            "context": context.get(e["id"], []), "labels": [],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--context_path", default="/home/jtan/adl_hw1/data/context.json")
    ap.add_argument("--test_path", default="/home/jtan/adl_hw1/data/test.json")
    ap.add_argument("--gold_path", default="/home/jtan/adl_hw1/data/public_test_gold.csv")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run_name)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, "thresholds.json")) as f:
        th = json.load(f)
    thresholds = np.array([th["thresholds"][l] for l in LABELS])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_id = MODEL_REGISTRY[config["model_key"]]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(LABELS), problem_type="multi_label_classification"
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    model.eval()

    test_examples = build_test_examples(args.context_path, args.test_path)
    ds = IntentDataset(test_examples, tokenizer, config["context_mode"], config["k"], config["max_length"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    logits = np.concatenate(all_logits)
    probs = sigmoid(logits)
    preds = (probs >= thresholds[None, :]).astype(int)

    out_csv = args.out_csv or os.path.join(run_dir, "public_test_prediction.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + LABELS)
        for e, row in zip(test_examples, preds):
            writer.writerow([e["id"]] + list(row))

    gold = {}
    with open(args.gold_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold[row["id"]] = [int(row[l]) for l in LABELS]
    gold_arr = np.array([gold[e["id"]] for e in test_examples])

    metrics_tuned = compute_f1(gold_arr, preds)
    preds_flat = (probs >= 0.5).astype(int)
    metrics_flat = compute_f1(gold_arr, preds_flat)

    zh_idx = [i for i, e in enumerate(test_examples) if e["language"] == "zh"]
    en_idx = [i for i, e in enumerate(test_examples) if e["language"] == "en"]
    metrics_zh = compute_f1(gold_arr[zh_idx], preds[zh_idx])
    metrics_en = compute_f1(gold_arr[en_idx], preds[en_idx])

    print("=== PUBLIC TEST — DIAGNOSTIC ONLY, thresholds/model selected on dev split, not this set ===")
    print(f"tuned-threshold  macro={metrics_tuned['macro_f1']:.4f} micro={metrics_tuned['micro_f1']:.4f}")
    print(f"flat-0.5         macro={metrics_flat['macro_f1']:.4f} micro={metrics_flat['micro_f1']:.4f}")
    print(f"zh (n={len(zh_idx)})  macro={metrics_zh['macro_f1']:.4f} micro={metrics_zh['micro_f1']:.4f}")
    print(f"en (n={len(en_idx)})  macro={metrics_en['macro_f1']:.4f} micro={metrics_en['micro_f1']:.4f}")

    with open(os.path.join(run_dir, "public_test_eval.json"), "w") as f:
        json.dump({
            "tuned": metrics_tuned, "flat_0.5": metrics_flat,
            "zh": metrics_zh, "en": metrics_en,
        }, f, indent=2)


if __name__ == "__main__":
    main()
