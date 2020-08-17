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
    def __init__(self, n_in, n_out, pars):
        super(Decode_block, self).__init__()

        self.dic  = pars
        self.activate = nn.LeakyReLU(0.2, inplace=True)

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

    def forward(self, x):
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

        self.conv_mu  = nn.Conv2d(dic['channels'][-1], dic['channels'][-1], 3, 1, 1)
        self.conv_var = nn.Conv2d(dic['channels'][-1], dic['channels'][-1], 3, 1, 1)

        ### Create Decoder Module
        in_channels = dic['channels'][-1]
        self.decode = []
        for out_channels in reversed(dic['channels']):
            self.decode.append(Decode_block(in_channels, out_channels, dic))
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
        for i, module in enumerate(self.encode):
            x = module(x)
        return self.reparameterize(x)

    def decoder(self, x):
        for i, module in enumerate(self.decode):
            x = module(x)
        return x

    def interpolate(self, out1, out2, lamb):
        return [out1[-1] + lamb * (out2[-1] - out1[-1])]

    def get_latent_var(self, x):
        return self.encoder(x)

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu(emb), self.conv_var(emb)
        eps = torch.FloatTensor(logvar.size()).normal_().cuda()
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        embed, mu, covar = self.encoder(x)
        img_recon     = self.decoder(embed)
        return img_recon, mu, covar
