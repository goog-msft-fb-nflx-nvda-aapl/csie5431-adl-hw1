import numpy as np
from sklearn.metrics import f1_score

from models import LABELS


def compute_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_class_f1": {LABELS[i]: float(per_class[i]) for i in range(len(LABELS))},
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
