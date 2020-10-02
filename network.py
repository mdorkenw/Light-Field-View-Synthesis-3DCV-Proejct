### Libraries
import torch.nn as nn, torch
from torch.nn import init
import torch.nn.functional as F


######################### Resnet Block down ##############################################################

class ResnetBlockDown(nn.Module):
    """ Downsampling the identity like this ignores most of the values?!
        Could potentially be improved by using larger kernel for downsizing, but this is how they did it.
    """
    def __init__(self, n_in, n_out, pars, stride_v=1, stride_r=1):
        super(ResnetBlockDown, self).__init__()
        #self.norm = pars['norm']
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm3d(n_in)

        self.left = [nn.Conv3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1),  self.activate]
        self.left = nn.Sequential(*self.left)
        
        # Use identity if possible, otherwise downsample with kernel 1x1x1
        if stride_v == 1 and stride_r == 1 and n_in == n_out:
            self.right = nn.Identity()
        else:
            self.right = nn.Conv3d(n_in, n_out, 1, stride=(stride_v, stride_r, stride_r), padding=0)

    def forward(self, x):
        x = self.bn(x)
        return self.left(x) + self.right(x)


######################### Resnet Block up ##############################################################

class UpSample(nn.Module):
    def __init__(self, stride_v=1, stride_r=1):
        """ Repeated tensor is cropped when the stack dimension is increased, probably only works for stride_v=2.
            Cropping is not really nice, but they do it like that as well.
        """
        super(UpSample, self).__init__()
        self.stride_v = stride_v
        self.stride_r = stride_r

    def forward(self, x):
        if self.stride_r != 1:
            x = torch.repeat_interleave(x, self.stride_r, 4)
            x = torch.repeat_interleave(x, self.stride_r, 3)
        if self.stride_v ==2:
            x = torch.repeat_interleave(x, self.stride_v, 2)[...,:-1,:,:]
        return x

class ResnetBlockUp(nn.Module):
    """ Uses repetition for identity upsampling.
    """
    def __init__(self, n_in, n_out, pars, stride_v=1, stride_r=1):
        super(ResnetBlockUp, self).__init__()
        self.norm = pars['norm']
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm3d(n_in)

        self.left = [nn.ConvTranspose3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1,
                                        output_padding=(0, stride_r-1, stride_r-1)),  self.activate]
        self.left = nn.Sequential(*self.left)
        
        # Use identity if possible, otherwise upsample with kernel 1x1x1
        if stride_v == 1 and stride_r == 1 and n_in == n_out:
            self.right = nn.Identity()
        else:
            self.right = [UpSample(stride_v, stride_r),
                          nn.ConvTranspose3d(n_in, n_out, 1, stride=(1, 1, 1), padding=0, output_padding=0)]
            self.right = nn.Sequential(*self.right)

    def forward(self, x):
        x = self.bn(x)
        return self.left(x) + self.right(x)


############# AUTOENCODER MODEL (UNET) ############################################################################
###################################################################################################################
class VAE(nn.Module):
    def __init__(self, dic):
        super(VAE, self).__init__()

        self.dic = dic
        self.use_VAE = dic['use_VAE']
        
        channels = dic['channels']
        strides_res = dic['strides_res']
        strides_stack = dic['strides_stack']
        
        ## Encoder
        self.encoder = []
        
        # First 3-Block
        in_channels = dic['in_channels'] # 3
        out_channels = channels[0]
        self.encoder.extend([ResnetBlockDown(in_channels, out_channels, dic),
                            ResnetBlockDown(out_channels, out_channels, dic),
                            ResnetBlockDown(out_channels, channels[1], dic, stride_v=1, stride_r=2)])
        in_channels = channels[1]
        
        # Rest
        for i, out_channels in enumerate(channels[2:]):
            self.encoder.extend([ResnetBlockDown(in_channels, in_channels, dic),
                                ResnetBlockDown(in_channels, in_channels, dic),
                                ResnetBlockDown(in_channels, out_channels, dic, stride_r=strides_res[i+1], stride_v=strides_stack[i+1])])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*self.encoder)
        
        # Variational part
        if self.use_VAE:
            self.conv_mu  = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
            self.conv_var = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        
        ### Decoder
        self.decoder = []
        in_channels = channels[-1]
        
        # All except last
        for i, out_channels in enumerate(reversed(channels[1:-1])):
            self.decoder.extend([ResnetBlockUp(in_channels, out_channels, dic, stride_r=strides_res[-(i + 1)], stride_v=strides_stack[-(i + 1)]),
                                ResnetBlockUp(out_channels, out_channels, dic),
                                ResnetBlockUp(out_channels, out_channels, dic)])
            in_channels = out_channels
        
        # Last 3-Block
        self.decoder.extend([ResnetBlockUp(out_channels, channels[0], dic, stride_r=2, stride_v=1),
                            ResnetBlockUp(channels[0], channels[0], dic),
                            ResnetBlockUp(channels[0], 3, dic)])
        
        self.decoder = nn.Sequential(*self.decoder)


        ### Initialize
        self.reset_params()
        print("Number of parameters in encoder", sum(p.numel() for p in self.encoder.parameters()))
        print("Number of parameters in decoder", sum(p.numel() for p in self.decoder.parameters()))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def encode(self, x):
        out = self.encoder(x)
        return out

    def decode(self, x):
        out = self.decoder(x)
        return out

    def reparameterize(self, x):
        """ Nut sure whether eval version is correct? Use mu instead of simply x?
        """
        mu, logvar = self.conv_mu(x), self.conv_var(x)
        eps = torch.FloatTensor(logvar.size()).normal_().to(self.dic['device'])
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu) if self.training else mu, mu, logvar

    def forward(self, x):
        latent = self.encoder(x)
        
        if self.use_VAE:
            emb, mu, logvar = self.reparameterize(latent)
        else:
            emb = latent
            mu, logvar = torch.ones(latent.size()).type(torch.FloatTensor), torch.ones(latent.size()).type(torch.FloatTensor)
        
        img_recon   = self.decoder(emb)
        return img_recon, mu, logvar
