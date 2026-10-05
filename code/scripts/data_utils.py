import json

from models import LABELS

LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def labels_to_vector(labels):
    vec = [0.0] * len(LABELS)
    for l in labels:
        vec[LABEL_TO_IDX[l]] = 1.0
    return vec


def filter_lang(examples, lang):
    if lang == "all":
        return examples
    return [e for e in examples if e["language"] == lang]


def build_context_text(context, k=None):
    turns = context if k is None else context[-k:]
    parts = []
    for turn in turns:
        role = "USER" if turn["role"] == "user" else "ASSISTANT"
        parts.append(f"[{role}] {turn['content']}")
    return "\n".join(parts)


def build_input_text(example, mode, k=2):
    utt = f"[USER] {example['utterance']}"
    if mode == "none":
        return utt
    if mode == "lastk":
        ctx = build_context_text(example["context"], k=k)
        return f"{ctx}\n{utt}" if ctx else utt
    if mode == "full":
        ctx = build_context_text(example["context"], k=None)
        return f"{ctx}\n{utt}" if ctx else utt
    raise ValueError(mode)


def tokenize_headtail(tokenizer, example, max_length, max_utt_tokens=160):
    utt_text = f"[USER] {example['utterance']}"
    utt_ids = tokenizer.encode(utt_text, add_special_tokens=False)
    if len(utt_ids) > max_utt_tokens:
        utt_ids = utt_ids[-max_utt_tokens:]

    ctx_text = build_context_text(example["context"], k=None)
    ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False) if ctx_text else []

    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - num_special - len(utt_ids)
    if budget < 0:
        budget = 0
        utt_ids = utt_ids[-max_length + num_special:] if max_length > num_special else []

    if len(ctx_ids) > budget:
        head_len = int(budget * 0.4)
        tail_len = budget - head_len
        ctx_ids = (ctx_ids[:head_len] + ctx_ids[-tail_len:]) if tail_len > 0 else ctx_ids[:head_len]

    combined = ctx_ids + utt_ids
    return tokenizer.prepare_for_model(
        combined, max_length=max_length, truncation=True, padding=False,
    )
