import numpy as np
import torch, os
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import random

""" Does not work with files in the train/test folders!!
"""

class dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode = 'train', return_mode = ''):
        """ Dataset class.
            Mode, i.e. train or test, cannot be changed after initializing.
            all_directions determines whether one random direction is returned, or all three (horizontal, vertical, diagonal).
            Dataset size is for now the number of times it should choose one random scene, maybe change later if we have more data.
        """
        if not mode in ['train','test']: raise NameError('Mode does not exist!') # Not really necessary, but why not
        if not mode in ['train','test']: raise NameError('Mode does not exist!')
        
        self.img_size = opt.Network['image_size'] # Output size of images
        self.mode = mode # train or test
        self.return_mode = return_mode if return_mode else opt.Dataloader['return_mode']
        if not self.return_mode in ['random','random_hor_vert','hor_vert','all']: raise NameError('Return mode does not exist!')
        self.flip, self.mirror, self.rotate = opt.Dataloader['flip'], opt.Dataloader['mirror'], opt.Dataloader['rotate']
        self.seed = np.random.randint(2147483647) # Initialize first random seed, only relevant if not using [] to get elements
        
        self.length = int(opt.Training[mode+'_size']) # Dataset size
        self.img_path = opt.Paths['img_path'] + mode + '/' # Path to all scenes
        self.scenes = os.listdir(self.img_path) # Different scenes
        
        if mode == 'train': # Random crop and jitter, further augmentation happens later
            self.transform = transforms.Compose([
                                                 transforms.RandomCrop(self.img_size),
                                                 transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.15),
                                                 ])
        else: # Only random crop
            self.transform = transforms.Compose([
                                                 transforms.RandomCrop(self.img_size),
                                                 ])
        self.to_tensor = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                             ]) # Transform to tensor and normalize

    def load_img(self, data_path):
        """ Load image from path, apply transformation and set seed beforehand. """
        img = Image.open(data_path)
        random.seed(self.seed)
        return self.transform(img)
    
    def get_image_path(self, scene, image):
        """ Return path to image from scene. """
        return self.img_path + self.scenes[scene] + "/" + 'input_Cam' + str(image).zfill(3) + ".png"
    
    def stack_to_tensor(self, stack):
        """ Turn stack of images into torch tensor. """
        return torch.stack([self.to_tensor(image) for image in stack], dim=0)
        #return torch.cat([self.to_tensor(image) for image in stack], dim=0) # Should it not be like this?
    
    def get_augmentation_bools(self):
        """ Get random bools for augmentation, if respective augmentation is used. """
        flip    =   bool(random.getrandbits(1)) if self.flip else False
        mirror  =   bool(random.getrandbits(1)) if self.mirror else False
        rotate  =   bool(random.getrandbits(1)) if self.rotate else False
        return flip, mirror, rotate
    
    ## Get stacks
    def get_horizontal(self, scene, to_tensor = True, row = 4):
        """ Get horizontal stack from scene, choose whether it is converted to tensor and which row should be taken. """
        stack = [self.load_img(self.get_image_path(scene, i + 9*row)) for i in range(9)]
        if to_tensor:   return self.stack_to_tensor(stack)
        else:           return stack
    
    def get_vertical(self, scene, to_tensor = True, column = 4):
        """ Get vertical stack from scene, choose whether it is converted to tensor and which column should be taken. """
        stack = [self.load_img(self.get_image_path(scene, i + column)) for i in range(0, 81, 9)]
        if to_tensor:   return self.stack_to_tensor(stack)
        else:           return stack
    
    def get_diagonal(self, scene, to_tensor = True, left_right = True):
        """ Get diagonal stack from scene, choose whether it is converted to tensor. """
        stack = [self.load_img(self.get_image_path(scene, i*10 if left_right else i*10 + (8-2*i))) for i in range(9)]
        if to_tensor:   return self.stack_to_tensor(stack)
        else:           return stack
    
    def get_all_directions(self, scene, to_tensor = True):
        """ Get horizontal, vertical and diagonal stack for scene. """
        return self.get_horizontal(scene, to_tensor), self.get_vertical(scene, to_tensor), self.get_diagonal(scene, to_tensor)
    
    def get_hor_vert(self, scene, to_tensor = True):
        """ Get horizontal and vertical stack for scene. """
        return self.get_horizontal(scene, to_tensor), self.get_vertical(scene, to_tensor)
    
    def get_random_direction(self, scene, to_tensor = True):
        """ Get stack with random direction. """
        direction = np.random.randint(3) # Choose whether vert, hor or diag
        if direction == 0:      return self.get_horizontal(scene, to_tensor)
        elif direction == 1:    return self.get_vertical(scene, to_tensor)
        else:                   return self.get_diagonal(scene, to_tensor)
    ####
    
    ## Augment stacks
    def stack_flip(self, stack_, invert = False):
        """ Flip (vertically) every image in stack, choose whether order should be reversed.
            Invert vertical stack, don't invert horizontal stack.
            Invert both diagonals, exchange them. """
        stack = [ImageOps.flip(image) for image in stack_]
        if invert:  return stack[::-1]
        else:       return stack
    
    def stack_mirror(self, stack_, invert = False):
        """ Mirror (horizontally) every image in stack, choose whether order should be reversed.
            Invert horizontal stack, don't invert vertical stack.
            Don't invert diagonals, exchange them. """
        stack = [ImageOps.mirror(image) for image in stack_]
        if invert:  return stack[::-1]
        else:       return stack
    
    def stack_rotate(self, stack_, invert = False):
        """ Rotate (90° to the right) every image in stack, choose whether order should be reversed.
            Invert horizontal, don't invert vertical, exchange them.
            Invert left-right diagonal, don't invert right-left diagonal, exchange them."""
        stack = [image.rotate(90) for image in stack_]
        if invert:  return stack[::-1]
        else:       return stack
    ####
    
    ## Get augmented stacks
    def get_hor_vert_augmented(self, scene, flip = False, mirror = False, rotate = False, to_tensor = True):
        """ Get horizontal and vertical stack for scene with augmentation. """
        hor, vert = self.get_horizontal(scene, to_tensor = False), self.get_vertical(scene, to_tensor = False)
        
        if flip:    hor, vert = self.stack_flip(hor, invert = False), self.stack_flip(vert, invert = True)
        if mirror:  hor, vert = self.stack_mirror(hor, invert = True), self.stack_mirror(vert, invert = False)
        if rotate:  vert, hor = self.stack_rotate(hor, invert = True), self.stack_rotate(vert, invert = False)
        
        if to_tensor:   return self.stack_to_tensor(hor), self.stack_to_tensor(vert)
        else:           return hor, vert
    
    def get_diagonal_augmented(self, scene, flip = False, mirror = False, rotate = False, to_tensor = True):
        """ Get left-right diagonal for scene with augmentation. """
        left_right, right_left = self.get_diagonal(scene, to_tensor = False, left_right = True), self.get_diagonal(scene, to_tensor = False, left_right = False)
        
        if flip:    right_left, left_right = self.stack_flip(left_right, invert = True), self.stack_flip(right_left, invert = True)
        if mirror:  right_left, left_right = self.stack_mirror(left_right, invert = False), self.stack_mirror(right_left, invert = False)
        if rotate:  right_left, left_right = self.stack_rotate(left_right, invert = True), self.stack_rotate(right_left, invert = False)
        
        if to_tensor:   return self.stack_to_tensor(left_right)
        else:           return left_right
    
    def get_all_directions_augmented(self, scene, flip = False, mirror = False, rotate = False, to_tensor = True):
        """ Get horizontal, vertical and diagonal stack for scene with augmentation. """
        hor, vert = self.get_hor_vert_augmented(scene, flip, mirror, rotate, to_tensor)
        diag = self.get_diagonal_augmented(scene, flip, mirror, rotate, to_tensor)
        return hor, vert, diag
    
    def get_random_direction_augmented(self, scene, flip = False, mirror = False, rotate = False, to_tensor = True):
        """ Get augmented stack with random direction. """
        direction = np.random.randint(3) # Choose whether vert, hor or diag
        if direction == 2:      return self.get_diagonal_augmented(scene, flip, mirror, rotate, to_tensor)
        else:                   return self.get_hor_vert_augmented(scene, flip, mirror, rotate, to_tensor)[direction]
    ####

    def __getitem__(self, idx):
        """ Return data. Chooses random scene from available ones. """
        self.seed = np.random.randint(2147483647) # Choose new random seed
        scene = np.random.randint(0, len(self.scenes)) # Choose random scene
        
        if self.return_mode == 'random': # Return only random direction
            if self.mode == 'train':    out = self.get_random_direction_augmented(scene, *self.get_augmentation_bools())
            else:                       out = self.get_random_direction(scene)
            return {'x': out}
            
        elif self.return_mode == 'random_hor_vert': # Return either horizontal or vertical
            if self.mode == 'train':    out = self.get_hor_vert_augmented(scene, *self.get_augmentation_bools())[np.random.randint(2)]
            else:                       out = self.get_hor_vert(scene)[np.random.randint(2)]
            return {'x': out}
            
        elif self.return_mode == 'hor_vert': # Return horizontal and vertical
            if self.mode == 'train':    hor, vert = self.get_hor_vert_augmented(scene, *self.get_augmentation_bools())
            else:                       hor, vert = self.get_hor_vert(scene)
            return {'horizontal': hor, 'vertical': vert}
            
        elif self.return_mode == 'all': # Return horizontal, vertical and diagonal
            if self.mode == 'train':    hor, vert, diag = self.get_all_directions_augmented(scene, *self.get_augmentation_bools())
            else:                       hor, vert, diag = self.get_all_directions(scene)
            return {'horizontal': hor, 'vertical': vert, 'diagonal': diag}

    def __len__(self):
        """ Return length. """
        return self.length

    def get_1hot_(self, label):
        hot1 = np.zeros(self.n_classes)
        hot1[label] = 1
        return hot1
