import argparse
import json
import os

import numpy as np

from eval_metrics import compute_f1, sigmoid
from models import LABELS

RESULTS_ROOT = "/home/jtan/adl_hw1/results"
GRID = np.arange(0.05, 0.96, 0.05)


def confusion_per_class(probs, labels, thresholds):
    preds = (probs >= thresholds[None, :]).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum(axis=0)
    fp = ((preds == 1) & (labels == 0)).sum(axis=0)
    fn = ((preds == 0) & (labels == 1)).sum(axis=0)
    return tp, fp, fn


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def tune_thresholds(probs, labels, micro_slack=0.02, n_passes=4):
    base_preds = (probs >= 0.5).astype(int)
    base_metrics = compute_f1(labels, base_preds)
    micro_floor = base_metrics["micro_f1"] - micro_slack

    thresholds = np.full(len(LABELS), 0.5)
    for _ in range(n_passes):
        for c in range(len(LABELS)):
            best_obj, best_t = -1e9, thresholds[c]
            for t in GRID:
                thresholds[c] = t
                tp, fp, fn = confusion_per_class(probs, labels, thresholds)
                per_class_f1 = [f1_from_counts(tp[i], fp[i], fn[i]) for i in range(len(LABELS))]
                macro = float(np.mean(per_class_f1))
                micro = f1_from_counts(tp.sum(), fp.sum(), fn.sum())
                obj = macro if micro >= micro_floor else macro - 10 * (micro_floor - micro)
                if obj > best_obj:
                    best_obj, best_t = obj, t
            thresholds[c] = best_t
    return thresholds, base_metrics, micro_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--micro_slack", type=float, default=0.02)
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run_name)
    logits = np.load(os.path.join(run_dir, "dev_logits.npy"))
    labels = np.load(os.path.join(run_dir, "dev_labels.npy"))
    probs = sigmoid(logits)

    thresholds, base_metrics, micro_floor = tune_thresholds(probs, labels, args.micro_slack)

    tuned_preds = (probs >= thresholds[None, :]).astype(int)
    tuned_metrics = compute_f1(labels, tuned_preds)

    print(f"flat-0.5   macro={base_metrics['macro_f1']:.4f} micro={base_metrics['micro_f1']:.4f}")
    print(f"tuned      macro={tuned_metrics['macro_f1']:.4f} micro={tuned_metrics['micro_f1']:.4f} (micro_floor={micro_floor:.4f})")
    for l in LABELS:
        idx = LABELS.index(l)
        print(f"  {l}: t={thresholds[idx]:.2f} f1@0.5={base_metrics['per_class_f1'][l]:.3f} f1@tuned={tuned_metrics['per_class_f1'][l]:.3f}")

    out = {
        "run_name": args.run_name,
        "thresholds": {LABELS[i]: float(thresholds[i]) for i in range(len(LABELS))},
        "flat_0.5": base_metrics,
        "tuned": tuned_metrics,
        "micro_floor": micro_floor,
    }
    with open(os.path.join(run_dir, "thresholds.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
