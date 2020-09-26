### Libraries
import torch.nn as nn, torch
from torch.nn import init
import torch.nn.functional as F


######################### Resnet Block down ##############################################################

class ResnetBlock_down(nn.Module):
    def __init__(self, n_in, n_out, pars, stride_v=1, stride_r=1):
        super(ResnetBlock_down, self).__init__()
        self.norm = pars['norm']
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm3d(n_in)

        self.left = [nn.Conv3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1),  self.activate]
        self.left = nn.Sequential(*self.left)

        self.right = nn.Conv3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1)

    def forward(self, x):
        x = self.bn(x)
        return self.left(x) + self.right(x)


######################### Resnet Block up ##############################################################

class ResnetBlock_up(nn.Module):
    def __init__(self, n_in, n_out, pars, stride_v=1, stride_r=1):
        super(ResnetBlock_up, self).__init__()
        self.norm = pars['norm']
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm3d(n_in)

        self.left = [nn.ConvTranspose3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1,
                                        output_padding=(0, stride_r-1, stride_r-1)),  self.activate]
        self.left = nn.Sequential(*self.left)

        self.right = nn.ConvTranspose3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r),  padding=1,
                                        output_padding=(0, stride_r-1, stride_r-1))

    def forward(self, x):
        x = self.bn(x)
        return self.left(x) + self.right(x)


############# AUTOENCODER MODEL (UNET) ############################################################################
###################################################################################################################
class VAE(nn.Module):
    def __init__(self, dic):
        super(VAE, self).__init__()

        self.dic    = dic
        ## Encoder for posture
        self.encode = []
        strides_r = [2, 1, 2, 2, 1, 2]
        strides_v = [1, 2, 1, 1, 2, 1]
        in_channels = dic['in_channels']
        out_channels = dic['channels'][0]
        self.encode.extend([ResnetBlock_down(in_channels, out_channels, dic),
                            ResnetBlock_down(out_channels, out_channels, dic),
                            ResnetBlock_down(out_channels, dic['channels'][1], dic, stride_v=1, stride_r=2)])
        in_channels = dic['channels'][1]
        for i, out_channels in enumerate(dic['channels'][2:]):
            self.encode.extend([ResnetBlock_down(in_channels, in_channels, dic),
                                ResnetBlock_down(in_channels, in_channels, dic),
                                ResnetBlock_down(in_channels, out_channels, dic, stride_r=strides_r[i+1], stride_v=strides_v[i+1])])
            in_channels = out_channels
        self.encode = nn.Sequential(*self.encode)

        self.conv_mu  = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.conv_var = nn.Conv3d(out_channels, out_channels, 3, 1, 1)

        ### Create Decoder Module
        self.decode = []
        in_channels = dic['channels'][-1]
        for i, out_channels in enumerate(reversed(dic['channels'][1:-1])):
            self.decode.extend([ResnetBlock_up(in_channels, in_channels, dic, stride_r=strides_r[-(i + 1)], stride_v=strides_v[-(i + 1)]),
                                ResnetBlock_up(in_channels, in_channels, dic),
                                ResnetBlock_up(in_channels, out_channels, dic)])
            in_channels = out_channels
        self.decode.extend([ResnetBlock_up(out_channels, out_channels, dic, stride_r=2, stride_v=1),
                            ResnetBlock_up(out_channels, out_channels, dic), ResnetBlock_up(out_channels, 3, dic)])

        self.decode = nn.Sequential(*self.decode)

        self.reset_params()
        print("Number of parameters in encoder", sum(p.numel() for p in self.encode.parameters()))
        print("Number of parameters in decoder", sum(p.numel() for p in self.decode.parameters()))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def encoder(self, x):
        for i, module in enumerate(self.encode):
            # if i == 0 or i == 1 :
            #     breakpoint()
            x = module(x)

        return self.reparameterize(x)

    def decoder(self, x,):
        for i, module in enumerate(self.decode):
            x = module(x)
        return x

    def get_latent_var(self, x):
        return self.encoder(x)

    def reparameterize(self, x):
        mu, logvar = self.conv_mu(x), self.conv_var(x)
        eps = torch.FloatTensor(logvar.size()).normal_().cuda()
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu) if self.training else x, mu, logvar

    def forward(self, x):
        emb, mu, logvar = self.encoder(x)
        img_recon   = self.decoder(emb)
        return img_recon, mu, logvar
