import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import build_input_text, labels_to_vector, load_json, tokenize_headtail
from eval_metrics import sigmoid
from models import LABELS


class InferDataset(Dataset):
    def __init__(self, examples, tokenizer, context_mode, k, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_mode = context_mode
        self.k = k
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        if self.context_mode == "headtail":
            feats = tokenize_headtail(self.tokenizer, e, self.max_length)
        else:
            text = build_input_text(e, self.context_mode, self.k)
            feats = self.tokenizer(text, truncation=True, max_length=self.max_length)
        feats["labels"] = labels_to_vector([])
        return feats


def collate(batch, tokenizer):
    for b in batch:
        b.pop("labels")
    return tokenizer.pad(batch, return_tensors="pt")


def load_route(model_dir, device):
    with open(f"{model_dir}/inference_meta.json") as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype="auto"
    ).to(device)
    model.eval()
    thresholds = np.array([meta["thresholds"][l] for l in LABELS])
    return model, tokenizer, meta, thresholds


def predict_route(model, tokenizer, meta, thresholds, examples, device, batch_size=32):
    if not examples:
        return {}
    ds = InferDataset(examples, tokenizer, meta["context_mode"], meta["k"], meta["max_length"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
    logits = np.concatenate(all_logits)
    probs = sigmoid(logits)
    preds = (probs >= thresholds[None, :]).astype(int)
    return {e["id"]: pred for e, pred in zip(examples, preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("context_path")
    ap.add_argument("test_path")
    ap.add_argument("out_path")
    ap.add_argument("--model_dir", default="./models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context = load_json(args.context_path)
    test = load_json(args.test_path)
    examples = []
    for e in test:
        examples.append({
            "id": e["id"], "language": e["language"], "utterance": e["utterance"],
            "context": context.get(e["id"], []),
        })

    zh_examples = [e for e in examples if e["language"] == "zh"]
    en_examples = [e for e in examples if e["language"] != "zh"]

    zh_model, zh_tok, zh_meta, zh_th = load_route(f"{args.model_dir}/zh", device)
    en_model, en_tok, en_meta, en_th = load_route(f"{args.model_dir}/en", device)

    preds = {}
    preds.update(predict_route(zh_model, zh_tok, zh_meta, zh_th, zh_examples, device))
    preds.update(predict_route(en_model, en_tok, en_meta, en_th, en_examples, device))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    with open(args.out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id"] + LABELS)
        for e in test:
            writer.writerow([e["id"]] + list(preds[e["id"]]))


if __name__ == "__main__":
    main()
