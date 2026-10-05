import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


def load_multiwoz_text():
    texts = []
    for split in ["train", "validation"]:
        ds = load_dataset("multi_woz_v22", split=split, trust_remote_code=True)
        for ex in ds:
            texts.extend(ex["turns"]["utterance"])
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="roberta-base")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model).to(device)

    texts = load_multiwoz_text()
    print(f"loaded {len(texts)} unlabeled MultiWOZ utterances (English, unsupervised MLM only)")

    enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
    examples = [{"input_ids": ids} for ids in enc["input_ids"]]

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        print(f"epoch={epoch} mlm_loss={total_loss / len(loader):.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    base_model = getattr(model, model.base_model_prefix)
    base_model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"saved domain-adapted base encoder to {args.out_dir}")


if __name__ == "__main__":
    main()
