import json
import os

import pandas as pd

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


def load_json_safe(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    rows = []
    for run_name in sorted(os.listdir(RESULTS_ROOT)):
        run_dir = os.path.join(RESULTS_ROOT, run_name)
        if not os.path.isdir(run_dir):
            continue
        config = load_json_safe(f"{run_dir}/config.json")
        thresholds = load_json_safe(f"{run_dir}/thresholds.json")
        public_eval = load_json_safe(f"{run_dir}/public_test_eval.json")
        if config is None:
            continue

        row = {
            "run_name": run_name,
            "model_key": config.get("model_key"),
            "hf_id": config.get("hf_id"),
            "context_mode": config.get("context_mode"),
            "k": config.get("k"),
            "loss": config.get("loss"),
            "lr": config.get("lr"),
            "epochs": config.get("epochs"),
            "warmup_ratio": config.get("warmup_ratio"),
            "lang_subset": config.get("lang_subset"),
            "seed": config.get("seed"),
            "n_train": config.get("n_train"),
            "n_dev": config.get("n_dev"),
            "best_epoch": config.get("best_epoch"),
            "dev_flat_macro": config.get("best_dev_macro_f1_at_0.5"),
        }
        if thresholds:
            row["dev_tuned_macro"] = thresholds.get("tuned", {}).get("macro_f1")
            row["dev_tuned_micro"] = thresholds.get("tuned", {}).get("micro_f1")
            row["dev_flat_macro"] = thresholds.get("flat_0.5", {}).get("macro_f1", row["dev_flat_macro"])
            row["dev_flat_micro"] = thresholds.get("flat_0.5", {}).get("micro_f1")
            row["dev_per_class_f1"] = json.dumps(thresholds.get("tuned", {}).get("per_class_f1"))
        if public_eval:
            row["public_tuned_macro"] = public_eval.get("tuned", {}).get("macro_f1")
            row["public_tuned_micro"] = public_eval.get("tuned", {}).get("micro_f1")
            if "zh" in public_eval:
                row["public_zh_macro"] = public_eval["zh"].get("macro_f1")
                row["public_zh_micro"] = public_eval["zh"].get("micro_f1")
            if "en" in public_eval:
                row["public_en_macro"] = public_eval["en"].get("macro_f1")
                row["public_en_micro"] = public_eval["en"].get("micro_f1")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = "/home/jtan/adl_hw1/results/results_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df[["run_name", "model_key", "lang_subset", "context_mode", "loss", "lr",
              "dev_tuned_macro", "dev_tuned_micro", "public_tuned_macro", "public_tuned_micro"]].to_string())


if __name__ == "__main__":
    main()
