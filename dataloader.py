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

        self.img_path = opt.Paths['img_path']
        self.img_size = opt.Network['image_size']
        self.mode = mode

        if mode == 'train':
            self.length   = int(opt.Training['train_size'])
            self.scenes   = os.listdir(self.img_path + 'training/')
            self.img_path = opt.Paths['img_path'] + 'training/'
        elif mode == 'test':
            self.length = int(opt.Training['test_size'])
            self.scenes   = os.listdir(self.img_path + mode)
            self.img_path = opt.Paths['img_path'] + 'test/'

        self.augment_train = transforms.Compose([
             transforms.Resize(self.img_size),
             transforms.RandomHorizontalFlip(p=0.5),
             RandomRightAngleRotation(),
             transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.15),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.augment_test  = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def get_1hot_(self, label):
        hot1 = np.zeros(self.n_classes)
        hot1[label] = 1
        return hot1

    def load_img(self, data_path, seed):
        img = Image.open(data_path)
        if self.mode == 'train':
            random.seed(seed)
            return self.augment_train(img)
        else:
            return self.augment_test(img)

    def __getitem__(self, idx):

        seed = np.random.randint(2147483647)

        rand_scene = np.random.randint(0, len(self.scenes))
        choice  = np.random.choice(3, 1)[0]

        x1 = [self.load_img(self.img_path + self.scenes[rand_scene] + "/" + 'input_Cam' + str(i + 4).zfill(3)
                            + ".png", seed) for i in range(0, 81, 9)]
        x2 = [self.load_img(self.img_path + self.scenes[rand_scene] + "/" + 'input_Cam' + str(i + 36).zfill(3)
                            + ".png", seed) for i in range(9)]
        x3 = [self.load_img(self.img_path + self.scenes[rand_scene] + "/" + 'input_Cam' + str(i * 10).zfill(3)
                            + ".png", seed) for i in range(9)]

        return {'x1': torch.cat(x1, dim=0), 'x2': torch.cat(x2, dim=0), 'x3': torch.cat(x3, dim=0)}

    def __len__(self):
        return self.length