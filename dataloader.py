import numpy as np
import torch, os
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

class RandomRightAngleRotation:
    """ Rotate by a multiple of 90°. Only behaves sensibly for square images! """

    def __init__(self):
        self.angles = [-90, 0, 90, 180]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train'):

        self.img_size = opt.Network['image_size']
        self.mode = mode
        
        self.length = int(opt.Training[mode+'_size'])
        self.img_path = opt.Paths['img_path'] + mode + '/'
        self.scenes = os.listdir(self.img_path)
        
        if mode == 'train':
            self.transform = transforms.Compose([
                 transforms.RandomCrop(self.img_size),
                 transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.15),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif mode == 'test':
            self.transform  = transforms.Compose([
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            raise NameError('Mode does not exist!')

    def load_img(self, data_path, seed):
        img = Image.open(data_path)
        random.seed(seed)
        return self.transform(img)
    
    def get_image_path(self, scene, image):
        return self.img_path + self.scenes[scene] + "/" + 'input_Cam' + str(image).zfill(3) + ".png"

    def __getitem__(self, idx):

        seed = np.random.randint(2147483647)

        rand_scene = np.random.randint(0, len(self.scenes))

        row = np.random.randint(3)

        if row == 0:
            out = [self.load_img(self.get_image_path(rand_scene, i + 4), seed) for i in range(0, 81, 9)]
        elif row == 1:
            out = [self.load_img(self.get_image_path(rand_scene, i + 36), seed) for i in range(9)]
        else:
            out = [self.load_img(self.get_image_path(rand_scene, i * 10), seed) for i in range(9)]
        return {'x': torch.stack(out, dim=0)}

    def __len__(self):
        return self.length

    def get_1hot_(self, label):
        hot1 = np.zeros(self.n_classes)
        hot1[label] = 1
        return hot1
