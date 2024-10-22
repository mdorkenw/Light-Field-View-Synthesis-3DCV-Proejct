import torch.nn as nn, torch
import torchvision

class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def KLDLoss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3)))

class Loss(nn.Module):
    def __init__(self, dic):
        super(Loss, self).__init__()
        self.w_kl = dic['w_kl']
        self.kl_th = dic['kl_threshold']
        self.use_kl = dic['use_kl']
        self.use_kl_threshold = dic['use_kl_threshold']
        self.recon_loss = dic['recon_loss']
        if not self.recon_loss in ['L1','L2']: raise NameError('Loss mode does not exist!')
    
    def reconstruction_loss(self, inp, target):
        if self.recon_loss == 'L1': L_recon = torch.mean(torch.abs(inp - target))
        elif self.recon_loss == 'L2': L_recon = torch.mean(torch.square(inp - target))
        return L_recon

    def forward(self, target, inp, mu, covar):

        L_kl = KLDLoss(mu, covar)
        w_kl = 0 if L_kl.item() < self.kl_th and self.use_kl_threshold else self.w_kl
        
        L_recon = self.reconstruction_loss(inp, target)
        
        L = L_recon + L_kl * w_kl if self.use_kl else L_recon
        
        return L, L_recon, L_kl
