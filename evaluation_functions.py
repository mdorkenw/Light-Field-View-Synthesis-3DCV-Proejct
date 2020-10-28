import numpy as np, os, sys, pandas as pd, ast, time, gc
import torch, torch.nn as nn, pickle as pkl, random
from tqdm import tqdm, trange
import network as net
import auxiliaries as aux
import loss as Loss
import dataloader as dloader
import argparse
from distutils.dir_util import copy_tree
from datetime import datetime
from torchvision import transforms
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from PIL import Image
from importlib import reload
import scipy.optimize as optimize
from pathlib import Path
import matplotlib.pyplot as plt

########## Fundamentals

def load_network_stuff():
    opt = aux.extract_setup_info('setup_valen.txt')[0] # Add your own setup file here
    gpu = opt.Training['GPU'] if torch.cuda.is_available() else []
    device = torch.device('cuda:{}'.format(gpu[0])) if gpu and torch.cuda.is_available() else torch.device('cpu')
    opt.Network['device'] = device
    
    network = net.VAE(opt.Network, opt).to(device)
    network.eval()
    print(opt.Paths['load_network_path'])
    save_dict = torch.load(opt.Paths['save_path'] + opt.Paths['load_network_path'], map_location=device)
    out = network.load_state_dict(save_dict['state_dict'])
    print(out)
    
    save_directory = os.path.dirname(opt.Paths['save_path'] + opt.Paths['load_network_path'])
    
    return opt, network, save_directory

def load_dict_stuff(name):
    opt = aux.extract_setup_info('setup_valen.txt')[0]
    save_directory = os.path.dirname(opt.Paths['save_path'] + opt.Paths['load_network_path'])
    dictionary = np.load(save_directory + '/save_generated/'+name+'.npy',allow_pickle='TRUE').item()
    
    mode = dictionary["mode"]
    idx = dictionary["idx"]
    ratios = dictionary["ratios"]
    angles = dictionary["angles"]
    generated_angles = dictionary["generated_angles"]
    generated_diagonal = dictionary["generated_diagonal"]
    positions = dictionary["positions"]
    positions_diagonal = dictionary["positions_diagonal"]
    
    dataset = dloader.dataset(opt, mode=mode, return_mode='all', img_size=512)
    horizontal, vertical, diagonal = dataset.get_all_directions(idx)
    grid_tensor = dataset.get_whole_grid(idx)[...,16:-16,16:-16]
    grid_tensor_full = dataset.get_whole_grid(idx)
    
    return opt, save_directory, mode, idx, ratios, angles, generated_angles, generated_diagonal, positions, positions_diagonal, horizontal, vertical, diagonal, grid_tensor, grid_tensor_full

def save_dict_stuff(mode, idx, ratios, angles, generated_angles, generated_diagonal, positions, positions_diagonal):
    dictionary = {"mode": mode,
                  "idx": idx,
                  "ratios": ratios,
                  "angles": angles,
                  "generated_angles": generated_angles,
                  "generated_diagonal": generated_diagonal,
                  "positions": positions,
                  "positions_diagonal": positions_diagonal}
    Path(save_directory + '/save_generated').mkdir(parents=True, exist_ok=True)
    np.save(save_directory + '/save_generated/generatedT'+mode[1:]+str(idx),dictionary)



########## Position Estimation

def get_min_difference(grid, image):
    difference = torch.abs(grid-image.unsqueeze(0).unsqueeze(0))
    error = torch.mean(difference,(2,3,4))
    x, y = (error.argmin()//grid.shape[0]).item(),(error.argmin()%grid.shape[0]).item()
    return error, x, y

def get_minimum(ys):
    if ys[0] > ys[2]: y = ys
    else: y = ys[::-1]
    
    d = (1-(y[2]-y[1])/(y[0]-y[1]))/2
    
    if ys[0] > ys[2]: return 1+d
    else: return 1-d

def get_minimum_edge(y):
    #if ys[0] < ys[2]: y = ys
    #else: y = ys[::-1]
    
    if y[1]-y[0] > y[2]-y[1]: return np.nan
    #if y[2] < y[1]: return np.nan
    
    d = (1-(y[1]-y[0])/(y[2]-y[1]))/2
    
    return d

def get_position_minimum(grid, image):
    error, x, y = get_min_difference(grid, image)
    
    if x == 0:
        x_ret = get_minimum_edge(np.asarray(error[0:3,y]))
    elif x == grid.shape[0]-1:
        x_ret = x-get_minimum_edge(np.asarray(error[-4:-1,y]))
    else:
        x_ret = x-1+get_minimum(np.asarray(error[x-1:x+2,y]))
    
    if y == 0:
        y_ret = get_minimum_edge(np.asarray(error[x,0:3]))
    elif y == grid.shape[1]-1:
        y_ret = y-get_minimum_edge(np.asarray(error[x,-4:-1]))
    else:
        y_ret = y-1+get_minimum(np.asarray(error[x,y-1:y+2]))
    
    return x_ret,y_ret

def get_positions(grid, images):
    if isinstance(images,list):
        positions = list()
        for stack in tqdm(images):
            stack_positions = list()
            for image in stack:
                stack_positions.append(get_position_minimum(grid,image))
            positions.append(stack_positions)
        return positions
    else:
        positions = list()
        for image in images:
            positions.append(get_position_minimum(grid,image))
        return positions

def get_original_positions(grid, tensor, function = lambda x: x):
    positions = list()
    for stack in tensor:
        for image in stack:
            positions.append(get_position_minimum(grid,function(image)))
    return positions

##### Plotting

def plot_grid(n=9,raster=False):
    plt.figure(figsize=(8,8))
    if raster:
        plt.rc('axes', axisbelow=True)
        plt.grid()
    plt.scatter(np.meshgrid(np.linspace(0,8,n),np.linspace(0,8,n))[0].flatten(),
                np.meshgrid(np.linspace(0,8,n),np.linspace(0,8,n))[1].flatten(),c="black")
    plt.ylim([8.5,-0.5])
    plt.xlim([-0.5,8.5])

def plot_with_fit(positions, label):
    x, y = position_to_xy(positions)
    m, b = fit_line(x,y)
    x_ = np.linspace(-1,9,100)
    plt.plot(x_, m*x_+b)
    plt.scatter(x, y, label=label+"{:.0f}°".format(to_angle(m)))

def plot_first_fit(positions):
    x, y = first_position_to_xy(positions)
    m, b = fit_line(x,y)
    x_ = np.linspace(-1,9,100)
    plt.plot(x_, m*x_+b,c="black")

def plot_original_positions(positions, factor,label="Orig.",c="gray"):
    plt.scatter([x*factor for x, y in positions],[y*factor for x, y in positions],label=label,c=c)

##### Plot Helpers

def to_angle(m):
    angle = np.rad2deg(np.arctan(m))
    if angle < -45: angle = 180+angle
    return angle
def position_to_xy(positions):
    x, y = np.asarray([x for x, y in positions]), np.asarray([y for x, y in positions])
    not_nan = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(not_nan) != len(not_nan): print("Found NaN")
    return x[not_nan], y[not_nan]
def first_position_to_xy(positions):
    x = np.asarray([positions[ind][0][0] for ind in range(len(positions))])
    y = np.asarray([positions[ind][0][1] for ind in range(len(positions))])
    not_nan = np.isfinite(x) & np.isfinite(y)
    return x[not_nan], y[not_nan]
def fit_line(x,y):
    """ Polyfit cannot deal with vertical lines!
    """
    m, b = np.polyfit(x, y, 1)
    if np.abs(m) < 5: return m, b
    else:
        m_, b_ = np.polyfit(y, x, 1)
        return 1/m_, -b_/m_

##### Position Interpolation

def interpolation_error(tens,image,x,y):
    x, y = torch.tensor(x), torch.tensor(y)
    interpol = y*(x*(tens[1,1])+(1-x)*(tens[0,1]))+(1-y)*(x*(tens[1,0])+(1-x)*(tens[0,0]))
    err = torch.mean(torch.abs(interpol-image)).item()
    return err

def get_position_interpolation(grid, image):
    """ Does not work, seems to always prefer blurred image!
    """
    error, x, y = get_min_difference(grid, image)
    square = grid[x:x+2,:] if error[x+1,y]<error[x-1,y] else grid[x-1:x+1,:]
    square = square[:,y:y+2] if error[x,y+1]<error[x,y-1] else square[:,y-1:y+1]
    
    numb = 9
    errors_interpolation = np.zeros((numb,numb))
    for i, i_ in enumerate(np.linspace(0,1,numb)):
        for j, j_ in enumerate(np.linspace(0,1,numb)):
            errors_interpolation[i,j] = interpolation_error(square,image,i_,j_)
    
    x_, y_ = (errors_interpolation.argmin()//numb).item()/(numb-1),(errors_interpolation.argmin()%numb).item()/(numb-1)
    x_ret = x+x_ if error[x+1,y]<error[x-1,y] else x+x_-1
    y_ret = y+y_ if error[x,y+1]<error[x,y-1] else y+y_-1
    return x_ret,y_ret



########## Misc

def reverse_tensor(tensor,axis = 0):
    idx = [i for i in range(tensor.size(axis)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    inverted_tensor = tensor.index_select(axis, idx)
    return inverted_tensor

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, sigma, channels=3, kernel_size=5, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups,padding=2)