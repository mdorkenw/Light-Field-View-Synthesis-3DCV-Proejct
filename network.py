### Libraries
import torch.nn as nn, torch
from torch.nn import init
import torch.nn.functional as F
import auxiliaries as aux
from tqdm import tqdm
import dataloader as dloader


######################### Resnet Block down ##############################################################

class ResnetBlockDown(nn.Module):
    """ Downsampling the identity like this ignores most of the values?!
        Could potentially be improved by using larger kernel for downsizing, but this is how they did it.
    """
    def __init__(self, n_in, n_out, pars, stride_v=1, stride_r=1, activate=True):
        super(ResnetBlockDown, self).__init__()
        self.bn = nn.BatchNorm3d(n_in)
        
        if activate:
            self.activate = nn.LeakyReLU(0.2, inplace=True)
            self.left = [nn.Conv3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1), self.activate]
            self.left = nn.Sequential(*self.left)
        else:
            self.left = nn.Conv3d(n_in, n_out, 3, stride=(stride_v, stride_r, stride_r), padding=1)
        
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

######################### Variational Part ##############################################################

class VariationBlock(nn.Module):
    def __init__(self, n_in, n_out, style, dic):
        super(VariationBlock, self).__init__()
        self.style = style
        
        if self.style in ['conv', 'norm_conv']:
            if self.style in ['norm_conv']: self.bn = nn.BatchNorm3d(n_in)
            self.conv_mu  = nn.Conv3d(n_in, n_out, 3, 1, 1)
            self.conv_var = nn.Conv3d(n_in, n_out, 3, 1, 1)
        
        if self.style == 'res_block':
            self.conv_mu  = ResnetBlockDown(n_in, n_out, dic, stride_r=1, stride_v=1, activate=False)
            self.conv_var = ResnetBlockDown(n_in, n_out, dic, stride_r=1, stride_v=1, activate=False)

    def forward(self, x):
        if self.style in ['norm_conv']: x = self.bn(x)
        return self.conv_mu(x), self.conv_var(x)

############# AUTOENCODER MODEL (UNET) ############################################################################
###################################################################################################################
class VAE(nn.Module):
    """ VAE styles:
            conv:       simply convolutional layers
            norm_conv:  batchnorm and convolutions
            res_block:  ResnetBlocks without activation (batchnorm + convolution + residual)
    """
    def __init__(self, dic, opt):
        super(VAE, self).__init__()
        if not dic['VAE_style'] in ['conv','norm_conv','res_block']: raise NameError('VAE style does not exist!')

        self.dic = dic
        self.opt = opt
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
        
        # Variational part
        if self.use_VAE:
            #del self.encoder[-1]
            #self.conv_mu  = ResnetBlockDown(channels[-2], channels[-1], dic, stride_r=strides_res[-1], stride_v=strides_stack[-1], activate=False)
            #self.conv_var = ResnetBlockDown(channels[-2], channels[-1], dic, stride_r=strides_res[-1], stride_v=strides_stack[-1], activate=False)
            self.variation = VariationBlock(channels[-1], channels[-1], dic['VAE_style'], dic)
        
        self.encoder = nn.Sequential(*self.encoder)
            
        
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
        if self.use_VAE: print("Number of parameters in variation", sum(p.numel() for p in self.variation.parameters()))
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

    def encode_reparametrize(self, x):
        out = self.encoder(x)
        return self.reparameterize(out)

    def decode(self, x):
        out = self.decoder(x)
        return out

    def reparameterize(self, x):
        """ Not sure whether eval version is correct? Is that even necessary?
        """
        if self.use_VAE:
            mu, logvar = self.variation(x)
            eps = torch.FloatTensor(logvar.size()).normal_().to(self.dic['device'])
            std = logvar.mul(0.5).exp_()
            return eps.mul(std).add_(mu), mu, logvar #if self.training else mu, mu, logvar
        else:
            return x, torch.ones(x.size()).type(torch.FloatTensor), torch.ones(x.size()).type(torch.FloatTensor)

    def forward(self, x):
        latent = self.encoder(x)
        emb, mu, logvar = self.reparameterize(latent)
        img_recon   = self.decoder(emb)
        return img_recon, mu, logvar
    
    def pass_through_image(self, x):
        crops = aux.get_crops(x)
        loader = torch.utils.data.DataLoader(crops, batch_size=self.opt.Training['bs'], shuffle=False)
        
        generated = list()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(self.dic['device'])
                out = self.forward(batch)[0].to('cpu')
                generated.extend(torch.split(out,1,0))
        
        result = aux.crops_to_tensor(generated)
        return result
    
    def pass_through_image_sum(self, hor, vert):
        crops_hor = aux.get_crops(hor)
        loader_hor = torch.utils.data.DataLoader(crops_hor, batch_size=self.opt.Training['bs'], shuffle=False)
        crops_vert = aux.get_crops(vert)
        loader_vert = torch.utils.data.DataLoader(crops_vert, batch_size=self.opt.Training['bs'], shuffle=False)
        
        generated = list()
        with torch.no_grad():
            for batch in tqdm(zip(loader_hor,loader_vert)):
                hor, vert = batch[0].to(self.dic['device']), batch[1].to(self.dic['device'])
                latent_hor = self.encode_reparametrize(hor)[0]
                latent_vert = self.encode_reparametrize(vert)[0]
                out = self.decode((latent_hor+latent_vert)/2).to('cpu')
                generated.extend(torch.split(out,1,0))
        
        result = aux.crops_to_tensor(generated)
        return result
    
    def save_whole_test_image(self, idx, epoch=None):
        """ Pass through whole test image idx and save it.
        """
        dataset = dloader.dataset(self.opt, mode='test', return_mode='all', img_size=512)
        horizontal, vertical, diagonal = dataset.get_all_directions(idx)
        name = "Test"+str(idx) if epoch == None else "{:03d}_Test".format(epoch)+str(idx)
        
        aux.save_full_image_grid(horizontal, self.pass_through_image(horizontal), name+"_horizontal", self.opt)
        aux.save_full_image_grid(vertical, self.pass_through_image(vertical), name+"_vertical", self.opt)
        aux.save_full_image_grid(diagonal, self.pass_through_image(diagonal), name+"_diagonal", self.opt)
        aux.save_full_image_grid(diagonal, self.pass_through_image_sum(horizontal, vertical), name+"_diagonal_sum", self.opt)
