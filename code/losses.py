import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021, ICCV. https://arxiv.org/abs/2009.14119"""

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1 - p
        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = p_pos * targets + p_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            modulator = (1 - pt) ** gamma
            loss = modulator * (loss_pos + loss_neg)
        else:
            loss = loss_pos + loss_neg
        return -loss.sum(dim=-1).mean()


class DistributionBalancedLoss(nn.Module):
    """Class-balanced weighting (Cui et al. 2019) + negative-tolerant regularization
    (Wu et al. 2020 DB-Loss / Huang et al. 2021 CB-NTR, EMNLP, github.com/Roche/BalancedLossNLP).
    pos_counts: per-class positive example counts in the training set.
    """

    def __init__(self, pos_counts, n_total, beta=0.9999, ntr_alpha=0.3, eps=1e-8):
        super().__init__()
        pos_counts = torch.as_tensor(pos_counts, dtype=torch.float).clamp(min=1)
        effective_num = 1.0 - torch.pow(beta, pos_counts)
        cb_weight = (1.0 - beta) / effective_num
        cb_weight = cb_weight / cb_weight.mean()  # normalize so overall loss scale is comparable to plain BCE
        self.register_buffer("cb_weight", cb_weight)
        # NTR: per-class margin >= 0, larger for rarer classes, shifts the negative-term
        # decision boundary up so rare classes' overwhelming negatives contribute less loss.
        margin = ntr_alpha * torch.log((n_total - pos_counts) / pos_counts).clamp(min=0)
        self.register_buffer("margin", margin)
        self.eps = eps

    def forward(self, logits, targets):
        p_pos = torch.sigmoid(logits)
        p_neg_boundary = torch.sigmoid(logits - self.margin)

        loss_pos = targets * self.cb_weight * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - p_neg_boundary).clamp(min=self.eps))
        return -(loss_pos + loss_neg).sum(dim=-1).mean()


def get_loss_fn(name, pos_weight=None, pos_counts=None, n_total=None):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    if name == "db":
        return DistributionBalancedLoss(pos_counts, n_total)
    raise ValueError(name)
