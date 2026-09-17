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


def get_loss_fn(name, pos_weight=None):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(name)
