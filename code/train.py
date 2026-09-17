import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data_utils import build_input_text, filter_lang, labels_to_vector, load_jsonl, tokenize_headtail
from eval_metrics import compute_f1, sigmoid
from losses import get_loss_fn
from models import LABELS, MODEL_REGISTRY

RESULTS_ROOT = "/home/jtan/adl_hw1/results"


class IntentDataset(Dataset):
    def __init__(self, examples, tokenizer, mode, k, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.mode = mode
        self.k = k
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        if self.mode == "headtail":
            feats = tokenize_headtail(self.tokenizer, e, self.max_length)
        else:
            text = build_input_text(e, self.mode, self.k)
            feats = self.tokenizer(text, truncation=True, max_length=self.max_length)
        feats["labels"] = labels_to_vector(e["labels"])
        feats["language"] = e["language"]
        return feats


def collate(batch, tokenizer):
    labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.float)
    langs = [b.pop("language") for b in batch]
    padded = tokenizer.pad(batch, return_tensors="pt")
    padded["labels"] = labels
    padded["language"] = langs
    return padded


def run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            langs = batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return logits, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--context_mode", required=True, choices=["none", "lastk", "full", "headtail"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--loss", default="bce", choices=["bce", "weighted_bce", "asl"])
    ap.add_argument("--lang_subset", default="all", choices=["all", "zh", "en"])
    ap.add_argument("--balance_lang", action="store_true")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--train_path", default="/home/jtan/adl_hw1/data/train_split.jsonl")
    ap.add_argument("--dev_path", default="/home/jtan/adl_hw1/data/dev_split.jsonl")
    ap.add_argument("--aug_path", default=None, help="extra train-only examples (e.g. back-translated), never added to dev")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_id = MODEL_REGISTRY[args.model_key]["hf_id"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(LABELS), problem_type="multi_label_classification"
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_examples = filter_lang(load_jsonl(args.train_path), args.lang_subset)
    if args.aug_path:
        aug_examples = filter_lang(load_jsonl(args.aug_path), args.lang_subset)
        print(f"adding {len(aug_examples)} augmented train-only examples from {args.aug_path}")
        train_examples = train_examples + aug_examples
    dev_examples = filter_lang(load_jsonl(args.dev_path), args.lang_subset)

    train_ds = IntentDataset(train_examples, tokenizer, args.context_mode, args.k, args.max_length)
    dev_ds = IntentDataset(dev_examples, tokenizer, args.context_mode, args.k, args.max_length)

    if args.balance_lang:
        n_en = sum(1 for e in train_examples if e["language"] == "en")
        n_zh = len(train_examples) - n_en
        w_en = 1.0 / max(n_en, 1)
        w_zh = 1.0 / max(n_zh, 1)
        weights = [w_en if e["language"] == "en" else w_zh for e in train_examples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            collate_fn=lambda b: collate(b, tokenizer),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, tokenizer),
        )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer),
    )

    pos_weight = None
    if args.loss == "weighted_bce":
        n = len(train_examples)
        pos_counts = np.array([sum(1 for e in train_examples if l in e["labels"]) for l in LABELS])
        pos_counts = np.clip(pos_counts, 1, None)
        pos_weight = torch.tensor(np.sqrt((n - pos_counts) / pos_counts), dtype=torch.float).to(device)
    loss_fn = get_loss_fn(args.loss, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    run_dir = os.path.join(RESULTS_ROOT, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_macro = -1.0
    best_epoch = -1
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch.pop("language")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            out = model(**batch)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        logits, labels = run_eval(model, dev_loader, device)
        preds = (sigmoid(logits) >= 0.5).astype(int)
        metrics = compute_f1(labels, preds)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch} train_loss={epoch_loss / len(train_loader):.4f} "
            f"dev_macro_f1@0.5={metrics['macro_f1']:.4f} dev_micro_f1@0.5={metrics['micro_f1']:.4f} "
            f"elapsed={elapsed:.0f}s", flush=True,
        )
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_loader), **metrics})

        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            np.save(os.path.join(run_dir, "dev_logits.npy"), logits)
            np.save(os.path.join(run_dir, "dev_labels.npy"), labels)

    config = vars(args)
    config["hf_id"] = hf_id
    config["n_train"] = len(train_examples)
    config["n_dev"] = len(dev_examples)
    config["best_epoch"] = best_epoch
    config["best_dev_macro_f1_at_0.5"] = best_macro
    config["wall_time_s"] = time.time() - t0

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(RESULTS_ROOT, "runs.jsonl"), "a") as f:
        f.write(json.dumps(config) + "\n")

    print(f"DONE run_name={args.run_name} best_epoch={best_epoch} best_dev_macro_f1@0.5={best_macro:.4f}")


if __name__ == "__main__":
    main()
