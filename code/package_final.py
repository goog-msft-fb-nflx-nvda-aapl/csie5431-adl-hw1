import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import filter_lang, load_jsonl
from eval_metrics import sigmoid
from models import LABELS, MODEL_REGISTRY

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def _package_one(run_name, out_dir, fp16=False):
    with open(f"{RESULTS_ROOT}/{run_name}/config.json") as f:
        config = json.load(f)

    hf_id = MODEL_REGISTRY[config["model_key"]]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(LABELS), problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(f"{RESULTS_ROOT}/{run_name}/best_model.pt", map_location="cpu"))
    if fp16:
        model.half()

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return config


def package(run_name, out_dir, fp16=False):
    config = _package_one(run_name, out_dir, fp16=fp16)
    with open(f"{RESULTS_ROOT}/{run_name}/thresholds.json") as f:
        th = json.load(f)
    meta = {
        "run_name": run_name,
        "model_key": config["model_key"],
        "hf_id": MODEL_REGISTRY[config["model_key"]]["hf_id"],
        "context_mode": config["context_mode"],
        "k": config.get("k"),
        "max_length": config["max_length"],
        "thresholds": th["thresholds"],
    }
    with open(f"{out_dir}/inference_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"packaged {run_name} -> {out_dir}")


def package_ensemble(run_names, out_dir, fp16=False, lang_subset="en",
                      dev_path="/home/jtan/adl_hw1/data/dev_split.jsonl"):
    """Package multiple checkpoints for one route as an averaged-logit ensemble.
    Re-tunes thresholds fresh on the averaged dev probabilities (self-contained,
    doesn't depend on a prior ensemble_eval.py run's /tmp output)."""
    from eval_metrics import compute_f1
    from threshold_tune import tune_thresholds

    configs = []
    member_dirs = []
    for i, run_name in enumerate(run_names):
        member_dir = os.path.join(out_dir, f"member_{i}")
        config = _package_one(run_name, member_dir, fp16=fp16)
        configs.append(config)
        member_dirs.append(f"member_{i}")

    context_mode = configs[0]["context_mode"]
    max_length = configs[0]["max_length"]
    k = configs[0].get("k")
    assert all(c["context_mode"] == context_mode and c["max_length"] == max_length for c in configs), \
        "all ensemble members must share the same context_mode/max_length"

    dev_examples = filter_lang(load_jsonl(dev_path), lang_subset)
    dev_labels = np.array([[1.0 if l in e["labels"] else 0.0 for l in LABELS] for e in dev_examples])
    dev_probs_list = []
    for run_name in run_names:
        logits = np.load(f"{RESULTS_ROOT}/{run_name}/dev_logits.npy")
        dev_probs_list.append(sigmoid(logits))
    avg_dev_probs = np.mean(dev_probs_list, axis=0)
    thresholds, base_metrics, _ = tune_thresholds(avg_dev_probs, dev_labels)
    tuned_preds = (avg_dev_probs >= thresholds[None, :]).astype(int)
    tuned_metrics = compute_f1(dev_labels, tuned_preds)
    print(f"re-tuned ensemble thresholds on dev: macro={tuned_metrics['macro_f1']:.4f} micro={tuned_metrics['micro_f1']:.4f}")

    meta = {
        "run_names": run_names,
        "member_dirs": member_dirs,
        "model_key": configs[0]["model_key"],
        "hf_id": MODEL_REGISTRY[configs[0]["model_key"]]["hf_id"],
        "context_mode": context_mode,
        "k": k,
        "max_length": max_length,
        "thresholds": {LABELS[i]: float(thresholds[i]) for i in range(len(LABELS))},
    }
    with open(f"{out_dir}/ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"packaged {len(run_names)}-way ensemble {run_names} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", nargs="+", required=True, help="one run name, or several for an ensemble")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lang_subset", default="en", help="only used for ensemble threshold re-tuning")
    args = ap.parse_args()
    if len(args.run_name) == 1:
        package(args.run_name[0], args.out_dir, fp16=args.fp16)
    else:
        package_ensemble(args.run_name, args.out_dir, fp16=args.fp16, lang_subset=args.lang_subset)
