import argparse
import csv
import json

import numpy as np

from eval_metrics import compute_f1
from models import LABELS

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def load_pred_csv(path):
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["id"]] = [int(row[l]) for l in LABELS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh_run", required=True)
    ap.add_argument("--en_run", required=True)
    ap.add_argument("--test_path", default="/home/jtan/adl_hw1/data/test.json")
    ap.add_argument("--gold_path", default="/home/jtan/adl_hw1/data/public_test_gold.csv")
    args = ap.parse_args()

    zh_preds = load_pred_csv(f"{RESULTS_ROOT}/{args.zh_run}/public_test_prediction.csv")
    en_preds = load_pred_csv(f"{RESULTS_ROOT}/{args.en_run}/public_test_prediction.csv")

    test = json.load(open(args.test_path))
    gold = {}
    with open(args.gold_path) as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = [int(row[l]) for l in LABELS]

    routed_preds, gold_arr = [], []
    for e in test:
        pred = zh_preds[e["id"]] if e["language"] == "zh" else en_preds[e["id"]]
        routed_preds.append(pred)
        gold_arr.append(gold[e["id"]])

    routed_preds = np.array(routed_preds)
    gold_arr = np.array(gold_arr)
    metrics = compute_f1(gold_arr, routed_preds)
    print(f"=== ROUTED (zh={args.zh_run}, en={args.en_run}) — public test, diagnostic ===")
    print(f"macro={metrics['macro_f1']:.4f} micro={metrics['micro_f1']:.4f}")
    print(json.dumps(metrics["per_class_f1"], indent=2))


if __name__ == "__main__":
    main()
