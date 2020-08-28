import torch.nn as nn
import torch

def KLDLoss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3)))

class Loss(nn.Module):
    def __init__(self, dic):
        super(Loss, self).__init__()
        self.w_kl = dic['w_kl']

    def forward(self, inp, target, mus, covars):

        L_kl = 0
        for mu, covar in zip(mus, covars):
            L_kl += KLDLoss(mu, covar)
        L_kl /= len(mus)

        L_recon = torch.mean((inp - target) **2)

        Loss = L_kl * self.w_kl + L_recon

        return Loss, L_recon, L_kl
