import numpy as np
import torch.nn as nn
from torchvision import transforms


class Base_Loss(nn.Module):
    def __init__(self, dic):
        super(Base_Loss, self).__init__()
        self.n_classes = dic['n_classes']
        if self.n_classes > 1:
            self.loss    = nn.CrossEntropyLoss()
        else:
            self.loss    = nn.BCEWithLogitsLoss()

    def forward(self, inp, target):
        if self.n_classes > 1:
            return self.loss(inp, target.long().reshape(-1))
        else:
            return self.loss(inp.reshape(-1), target.reshape(-1))


class LabelSmoothing(nn.Module):
    def __init__(self, dic):
        super(LabelSmoothing, self).__init__()
        smoothing = dic['smoothing']
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target, mode='train'):
        if mode == 'train':
            x = x.float()
            target = target.float()
            logprobs = nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return nn.functional.cross_entropy(x, target.long())

