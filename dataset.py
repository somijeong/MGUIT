import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
from torchvision.transforms import functional as TF
import torch

class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.size_h = opts.resize_size_h
        self.size_w = opts.resize_size_w
        self.crop_size = opts.crop_size
        self.mask_size = 90
        # A
        dir_A = os.path.join(self.dataroot, 'label', opts.input_a + '_' + opts.phase)
        list_A = make_dataset(dir_A)
        self.list_A = sorted(list_A)

        # B
        dir_B = os.path.join(self.dataroot, 'label', opts.input_b + '_' + opts.phase)
        list_B = make_dataset(dir_B)
        self.list_B = sorted(list_B)

        self.A_size = len(self.list_A)
        self.B_size = len(self.list_B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b

        # setup image transformation
        transforms = [Resize((self.mask_size, self.mask_size), Image.NEAREST),
                      ToTensor()]
        transforms_ = [ToTensor(),
                       Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transforms_m = Compose(transforms)
        self.transforms_i = Compose(transforms_)
        print('A: %d, B: %d images'%(self.A_size, self.B_size))
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            name_A = np.loadtxt(self.list_A[index], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
            cls_ = np.loadtxt(self.list_A[index], dtype=int, delimiter=' ', skiprows=2, usecols=(0))
            box_ = np.loadtxt(self.list_A[index], dtype=float, delimiter=' ', skiprows=2, usecols=(1,2,3,4))
            if box_.size == 4:
                cls_ = np.expand_dims(cls_, axis=0)
                box_ = np.expand_dims(box_, axis=0)
            mask_ = self.make_mask(cls_, box_)
            data_A, mask_A = self.load_img(os.path.join(self.dataroot, name_A[0]), self.input_dim_A, mask_)

            index1 = random.randint(0, self.B_size - 1)
            name_B = np.loadtxt(self.list_B[index1], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
            cls_ = np.loadtxt(self.list_B[index1], dtype=int, delimiter=' ', skiprows=2, usecols=(0))
            box_ = np.loadtxt(self.list_B[index1], dtype=float, delimiter=' ', skiprows=2, usecols=(1,2,3,4))
            if box_.size == 4:
                cls_ = np.expand_dims(cls_, axis=0)
                box_ = np.expand_dims(box_, axis=0)
            mask_ = self.make_mask(cls_, box_)
            data_B, mask_B = self.load_img(os.path.join(self.dataroot, name_B[0]), self.input_dim_B, mask_)
        else:
            index1 = random.randint(0, self.A_size - 1)
            name_A = np.loadtxt(self.list_A[index1], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
            cls_ = np.loadtxt(self.list_A[index1], dtype=int, delimiter=' ', skiprows=2, usecols=(0))
            box_ = np.loadtxt(self.list_A[index1], dtype=float, delimiter=' ', skiprows=2, usecols=(1,2,3,4))
            if box_.size == 4:
                cls_ = np.expand_dims(cls_, axis=0)
                box_ = np.expand_dims(box_, axis=0)
            mask_ = self.make_mask(cls_, box_)
            data_A, mask_A = self.load_img(os.path.join(self.dataroot, name_A[0]), self.input_dim_A, mask_)

            name_B = np.loadtxt(self.list_B[index], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
            cls_ = np.loadtxt(self.list_B[index], dtype=int, delimiter=' ', skiprows=2, usecols=(0))
            box_ = np.loadtxt(self.list_B[index], dtype=float, delimiter=' ', skiprows=2, usecols=(1,2,3,4))
            if box_.size == 4:
                cls_ = np.expand_dims(cls_, axis=0)
                box_ = np.expand_dims(box_, axis=0)
            mask_ = self.make_mask(cls_, box_)
            data_B, mask_B = self.load_img(os.path.join(self.dataroot, name_B[0]), self.input_dim_B, mask_)

        return data_A, data_B, mask_A, mask_B

    def load_img(self, img_name, input_dim, mask):
        img = Image.open(img_name).convert('RGB')
        img, mask = self.randomcrop(img, mask)
        img = self.transforms_i(img)
        mask_ = self.transforms_m(mask[0])
        for i in range(1, len(mask)):
            mask_ = torch.cat((mask_, self.transforms_m(mask[i])), 0)
        s_mask = torch.sum(mask_, 0)
        m = torch.where(s_mask == 0, torch.ones(1, self.mask_size, self.mask_size),
                        torch.zeros(1, self.mask_size, self.mask_size))
        mask = torch.cat((m, mask_), 0)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img, mask

    def make_mask(self, cls, box):
        mask1 = Image.new('L', (self.size_w, self.size_h), 0)
        mask2 = Image.new('L', (self.size_w, self.size_h), 0)
        mask3 = Image.new('L', (self.size_w, self.size_h), 0)

        box_ = np.round(box / 4) * 4
        box_ = box_.astype('int')
        w = box_[:,2] - box_[:,0]
        h = box_[:,3] - box_[:,1]

        for idx in range(cls.size):
            if cls[idx] == 1:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask1.paste(sub, (box_[idx, 0], box_[idx, 1]))
            elif cls[idx] == 2:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask2.paste(sub, (box_[idx, 0], box_[idx, 1]))
            else: # cls[idx] == 3:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask3.paste(sub, (box_[idx, 0], box_[idx, 1]))
        mask = []
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)
        return mask

    def __len__(self):
        return self.dataset_size

    def randomcrop(self, im, mask):
        # Random crop
        i = random.randint(0, self.size_w - self.crop_size)
        im = TF.crop(im, 0, i, self.crop_size, self.crop_size)
        masks = []
        for i in range(len(mask)):
            masks.append(TF.crop(mask[i], 0, i, self.crop_size, self.crop_size))
        # Random horizontal flipping
        if random.random() > 0.5:
            im = TF.hflip(im)
            mask = masks
            masks = []
            for i in range(len(mask)):
                masks.append(TF.hflip(mask[i]))
        return im, masks

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images



class dataset_single_mask(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.size_h = opts.resize_size_h
        self.size_w = opts.resize_size_w
        self.crop_size = opts.crop_size
        self.mask_size_h = int(opts.resize_size_h/4)
        self.mask_size_w = int(opts.resize_size_w/4)

        if opts.a2b:
            dir = os.path.join(self.dataroot, 'label', opts.input_a + '_' + opts.phase)
            self.input_dim = opts.input_dim_a
        else:
            dir = os.path.join(self.dataroot, 'label', opts.input_b + '_' + opts.phase)
            self.input_dim = opts.input_dim_b

        list = make_dataset(dir)
        self.list = sorted(list[0:-1:5])

        self.dataset_size = len(self.list)
        # # setup image transformation
        transforms = [Resize((self.mask_size_h, self.mask_size_w), Image.NEAREST),
                      ToTensor()]
        transforms_ = [ToTensor(),
                       Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transforms_m = Compose(transforms)
        self.transforms_i = Compose(transforms_)
        print('test: %d images' % self.dataset_size)
        return

    def __getitem__(self, index):
        name_ = np.loadtxt(self.list[index], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
        cls_ = np.loadtxt(self.list[index], dtype=int, delimiter=' ', skiprows=2, usecols=(0))
        box_ = np.loadtxt(self.list[index], dtype=float, delimiter=' ', skiprows=2, usecols=(1,2,3,4))
        if box_.size == 4:
            cls_ = np.expand_dims(cls_, axis=0)
            box_ = np.expand_dims(box_, axis=0)
        mask_ = self.make_mask(cls_, box_)
        img, mask = self.load_img(os.path.join(self.dataroot, name_[0]), self.input_dim, mask_)
        filename = name_[0]
        filename = filename.replace('/', '_')[0:-4]
        return img, mask, filename

    def load_img(self, img_name, input_dim, mask):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms_i(img)
        pad = torch.zeros(img.size(0), img.size(1), 4)
        img = torch.cat([img, pad], dim=2)
        mask_ = self.transforms_m(mask[0])
        for i in range(1, len(mask)):
            mask_ = torch.cat((mask_, self.transforms_m(mask[i])), 0)
        s_mask = torch.sum(mask_, 0)
        m = torch.where(s_mask == 0, torch.ones(1, self.mask_size_h, self.mask_size_w),
                        torch.zeros(1, self.mask_size_h, self.mask_size_w))
        mask = torch.cat((m, mask_), 0)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img, mask

    def make_mask(self, cls, box):
        mask1 = Image.new('L', (self.size_w, self.size_h), 0)
        mask2 = Image.new('L', (self.size_w, self.size_h), 0)
        mask3 = Image.new('L', (self.size_w, self.size_h), 0)

        box_ = np.round(box / 4) * 4
        box_ = box_.astype('int')
        w = box_[:,2] - box_[:,0]
        h = box_[:,3] - box_[:,1]

        for idx in range(cls.size):
            if cls[idx] == 1:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask1.paste(sub, (box_[idx, 0], box_[idx, 1]))
            elif cls[idx] == 2:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask2.paste(sub, (box_[idx, 0], box_[idx, 1]))
            else: # cls[idx] == 3:
                sub = Image.new('L', (w[idx], h[idx]), 255)
                mask3.paste(sub, (box_[idx, 0], box_[idx, 1]))
        mask = []
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)
        return mask

    def __len__(self):
        return self.dataset_size


class dataset_single(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.size_h = opts.resize_size_h
        self.size_w = opts.resize_size_w

        if opts.a2b:
            dir = os.path.join(self.dataroot, 'label', opts.input_a + '_' + opts.phase)
            self.input_dim = opts.input_dim_a
        else:
            dir = os.path.join(self.dataroot, 'label', opts.input_b + '_' + opts.phase)
            self.input_dim = opts.input_dim_b

        list = make_dataset(dir)
        self.list = sorted(list[0:-1:5])

        self.dataset_size = len(self.list)

        transforms = [ToTensor(),
                       Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transforms = Compose(transforms)
        print('test: %d images' % self.dataset_size)
        return

    def __getitem__(self, index):
        name_ = np.loadtxt(self.list[index], dtype=str, delimiter=' ', skiprows=0, usecols=(0))
        img = Image.open(os.path.join(self.dataroot, name_[0])).convert('RGB')
        img = self.transforms(img)
        pad = torch.zeros(img.size(0), img.size(1), 4)
        img = torch.cat([img, pad], dim=2)
        filename = name_[0]
        filename = filename.replace('/', '_')[0:-4]
        return img, filename

    def __len__(self):
        return self.dataset_size