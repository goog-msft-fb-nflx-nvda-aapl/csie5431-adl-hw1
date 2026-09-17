import argparse
import random

from data_utils import load_jsonl
from models import LABELS


def imbalance_score(subset, full_label_rate, full_lang_rate):
    n = len(subset)
    label_rate = {l: sum(1 for e in subset if l in e["labels"]) / n for l in LABELS}
    lang_rate = sum(1 for e in subset if e["language"] == "en") / n
    score = sum(abs(label_rate[l] - full_label_rate[l]) for l in LABELS)
    score += abs(lang_rate - full_lang_rate)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="/home/jtan/adl_hw1/data/train.jsonl")
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--n_trials", type=int, default=500)
    ap.add_argument("--out_train", default="/home/jtan/adl_hw1/data/train_split.jsonl")
    ap.add_argument("--out_dev", default="/home/jtan/adl_hw1/data/dev_split.jsonl")
    args = ap.parse_args()

    examples = load_jsonl(args.train_path)
    n = len(examples)
    n_dev = int(n * args.dev_frac)

    full_label_rate = {l: sum(1 for e in examples if l in e["labels"]) / n for l in LABELS}
    full_lang_rate = sum(1 for e in examples if e["language"] == "en") / n

    best_score = None
    best_dev_idx = None
    for trial in range(args.n_trials):
        rng = random.Random(trial)
        idx = list(range(n))
        rng.shuffle(idx)
        dev_idx = set(idx[:n_dev])
        dev_subset = [examples[i] for i in dev_idx]
        score = imbalance_score(dev_subset, full_label_rate, full_lang_rate)
        if best_score is None or score < best_score:
            best_score = score
            best_dev_idx = dev_idx

    train_examples = [examples[i] for i in range(n) if i not in best_dev_idx]
    dev_examples = [examples[i] for i in best_dev_idx]

    with open(args.out_train, "w") as f:
        for e in train_examples:
            f.write(__import__("json").dumps(e, ensure_ascii=False) + "\n")
    with open(args.out_dev, "w") as f:
        for e in dev_examples:
            f.write(__import__("json").dumps(e, ensure_ascii=False) + "\n")

    print(f"n_train={len(train_examples)} n_dev={len(dev_examples)} best_imbalance_score={best_score:.4f}")
    for l in LABELS:
        tr_rate = sum(1 for e in train_examples if l in e["labels"]) / len(train_examples)
        dv_rate = sum(1 for e in dev_examples if l in e["labels"]) / len(dev_examples)
        print(f"  {l}: train={tr_rate:.3f} dev={dv_rate:.3f} full={full_label_rate[l]:.3f}")
    tr_en = sum(1 for e in train_examples if e["language"] == "en") / len(train_examples)
    dv_en = sum(1 for e in dev_examples if e["language"] == "en") / len(dev_examples)
    print(f"  lang_en: train={tr_en:.3f} dev={dv_en:.3f} full={full_lang_rate:.3f}")


if __name__ == "__main__":
    main()
