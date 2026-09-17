import argparse
import json

import torch
from transformers import MarianMTModel, MarianTokenizer

from data_utils import load_jsonl


def translate_batch(texts, tokenizer, model, device, max_length=128, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=max_length, num_beams=4)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = [e for e in load_jsonl(args.in_path) if e["language"] == args.lang]
    print(f"back-translating {len(examples)} {args.lang} examples")

    if args.lang == "en":
        fwd_id, back_id = "Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"
    else:
        fwd_id, back_id = "Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-en-zh"

    fwd_tok = MarianTokenizer.from_pretrained(fwd_id)
    fwd_model = MarianMTModel.from_pretrained(fwd_id).to(device).eval()
    back_tok = MarianTokenizer.from_pretrained(back_id)
    back_model = MarianMTModel.from_pretrained(back_id).to(device).eval()

    utterances = [e["utterance"] for e in examples]
    pivot = translate_batch(utterances, fwd_tok, fwd_model, device)
    roundtrip = translate_batch(pivot, back_tok, back_model, device)

    augmented = []
    for e, new_utt in zip(examples, roundtrip):
        if new_utt.strip() and new_utt.strip() != e["utterance"].strip():
            new_e = dict(e)
            new_e["id"] = e["id"] + "_bt"
            new_e["utterance"] = new_utt.strip()
            augmented.append(new_e)

    print(f"produced {len(augmented)} non-trivial paraphrases (of {len(examples)} inputs)")
    with open(args.out_path, "w") as f:
        for e in augmented:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
