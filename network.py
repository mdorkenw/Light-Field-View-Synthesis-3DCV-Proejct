### Libraries
import torch.nn as nn, torch
from torchvision import models
from collections import namedtuple
from torch.nn import init
import torch.nn.functional as F

from utils.normalization_layer import Norm2D as Norm

########################################  Upsample  ############################################
################################################################################################

class Upsample(nn.Module):
    def __init__(self, mode, scale_factor):
        super(Upsample, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, mode='bilinear', scale_factor=self.scale_factor, align_corners=True)
        return x


def up(in_channels, out_channels, mode='upsample'):
    if mode == 'transpose':
        return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
                nn.GroupNorm(4 * out_channels, num_groups=16),
                nn.LeakyReLU(0.2, inplace=True)]
    elif mode == 'bilinear':
        return [Upsample(mode='bilinear', scale_factor=2)]
    elif mode == 'subpixel':
        return [nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(4*out_channels, num_groups=16),
                nn.PixelShuffle(upscale_factor=2)]

#########################-- Encoder Block --#########################################################
#####################################################################################################
class Encode_block(nn.Module):
    def __init__(self, n_in, n_out, pars, depth=2):
        super(Encode_block, self).__init__()
        self.norm  = pars['norm']
        self.activate = nn.LeakyReLU(0.2, inplace=True)

        ## Down convolution
        self.down = [nn.Conv2d(n_in, n_out, 3, 2, 1), Norm(n_out, pars), self.activate]
        self.down = nn.Sequential(*self.down)

        self.layer = []
        for num in range(depth):
            self.layer.append(nn.Conv2d(n_out, n_out, 3, 1, 1))
            self.layer.append(Norm(n_out, pars))
            if num < depth - 1:
                self.layer.append(self.activate)

        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        x = self.down(x)
        return self.activate(self.layer(x) + x)

#########################-- Decoder Block --#########################################################
#####################################################################################################
class Decode_block(nn.Module):
    def __init__(self, n_in, n_out, pars, iter):
        super(Decode_block, self).__init__()

        self.dic  = pars
        self.skips = pars['skips']
        self.activate = nn.LeakyReLU(0.2, inplace=True)

        if self.skips:
            n_in += pars['channels'][-(1 + iter)]

        self.upsample = nn.Sequential(*up(n_in, n_in, mode=pars['up_mode']))

        self.resnet = []
        for num in range(pars['depth']):
            if num == 0:
                self.resnet.append(nn.Conv2d(n_in, n_out, 3, 1, 1))
            else:
                self.resnet.append(nn.Conv2d(n_out, n_out, 3, 1, 1))
            self.resnet.append(Norm(n_out, pars))
            if (num + 1) < pars['depth']:
                self.resnet.append(self.activate)

        self.resnet = nn.Sequential(*self.resnet)

        self.down = [nn.Conv2d(n_in, n_out, 1), Norm(n_out, pars)]
        self.down = nn.Sequential(*self.down)

    def forward(self, x, skip_connc):
        if self.skips:
            x = torch.cat((x, skip_connc), dim=1)
        x = self.upsample(x)
        return self.activate(self.resnet(x) + self.down(x))

############# AUTOENCODER MODEL (UNET) ############################################################################
###################################################################################################################
class VAE(nn.Module):
    def __init__(self, dic):
        super(VAE, self).__init__()

        self.dic    = dic
        ## Encoder for posture
        self.encode = []
        in_channels = dic['in_channels']
        for out_channels in dic['channels']:
            self.encode.append(Encode_block(in_channels, out_channels, dic, depth=dic['depth']))
            in_channels = out_channels
        self.encode = nn.Sequential(*self.encode)

        self.conv_mu  = nn.Sequential(*[nn.Conv2d(c, c, 3, 1, 1) for c in dic['channels']])
        self.conv_var = nn.Sequential(*[nn.Conv2d(c, c, 3, 1, 1) for c in dic['channels']])

        ### Create Decoder Module
        in_channels = dic['channels'][-1]
        self.decode = []
        for i, out_channels in enumerate(reversed(dic['channels'])):
            self.decode.append(Decode_block(in_channels, out_channels, dic, i))
            in_channels = out_channels
        self.decode.extend([nn.Conv2d(dic['channels'][0], dic['in_channels'], 3, 1, 1), nn.Tanh()])
        self.decode = nn.Sequential(*self.decode)

        self.reset_params()
        print("Number of parameters in encoder", sum(p.numel() for p in self.encode.parameters()))
        print("Number of parameters in decoder", sum(p.numel() for p in self.decode.parameters()))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def encoder(self, x):
        skips = []
        for i, module in enumerate(self.encode):
            x = module(x)
            skips.append(x)
        return self.reparameterize(skips)

    def decoder(self, x, skips):
        for i, module in enumerate(self.decode):
            try:
                x = module(x, skips[-(i + 1)])
            except IndexError:
                x = module(x)
        return x

    def interpolate(self, out1, out2):
        out = []
        for skip1, skip2 in zip(out1, out2):
            out.append(skip1 + 0.5 * (skip2 - skip1))
        return out

    def get_latent_var(self, x):
        return self.encoder(x)

    def reparameterize(self, skips):
        out, mus, vars = [], [], []
        for i, skip in enumerate(skips):
            mu, logvar = self.conv_mu[i](skip), self.conv_var[i](skip)
            eps = torch.FloatTensor(logvar.size()).normal_().cuda()
            std = logvar.mul(0.5).exp_()
            out.append(eps.mul(std).add_(mu)); mus.append(mu); vars.append(logvar)
        return out, mus, vars

    def forward(self, x1, x2):
        skip1, mu, covar = self.encoder(x1)
        skip2, mu, covar = self.encoder(x2)

        skips_inter = self.interpolate(skip1, skip2)
        img_recon   = self.decoder(skips_inter[-1], skips_inter)
        return img_recon, mu, covar
