import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models import LABELS, MODEL_REGISTRY

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def package(run_name, out_dir):
    with open(f"{RESULTS_ROOT}/{run_name}/config.json") as f:
        config = json.load(f)
    with open(f"{RESULTS_ROOT}/{run_name}/thresholds.json") as f:
        th = json.load(f)

    hf_id = MODEL_REGISTRY[config["model_key"]]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(LABELS), problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(f"{RESULTS_ROOT}/{run_name}/best_model.pt", map_location="cpu"))

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "run_name": run_name,
        "model_key": config["model_key"],
        "hf_id": hf_id,
        "context_mode": config["context_mode"],
        "k": config.get("k"),
        "max_length": config["max_length"],
        "thresholds": th["thresholds"],
    }
    with open(f"{out_dir}/inference_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"packaged {run_name} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    package(args.run_name, args.out_dir)
