import os
import csv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pdb import set_trace as st

class Q1Dataset_pic(Dataset):
    def __init__(self, opt_data, phase='train',root=None):
        self.phase = phase
        exp_type = opt_data['type']
        AVP = opt_data['AVP']
        norm = opt_data['is_norm']

        if root == None:
            self.root = '../dataset/q1_data'
        else:
            self.root = root
        if phase == 'train' or phase == 'val':
            self.root = os.path.join(self.root, 'train')
            self.label_root = os.path.join('../dataset/q1_data', 'train'+str(exp_type)+'.csv')
            assert os.path.exists(self.label_root)
        elif phase == 'test':
            self.root = os.path.join(self.root, 'test')
        assert os.path.exists(self.root)

        self.namelist = os.listdir(self.root)
        self.namelist = [item for item in self.namelist if item[-4:] == '.jpg' or item[-4:] == '.png']
        self.namelist.sort()
        if phase == 'train' or phase == 'val':
            with open(self.label_root, 'r') as f:
                reader = csv.reader(f)
                self.labellist = list(reader)[1:]
            assert (len(self.namelist) == len(self.labellist))

        if phase == 'train':
            namelist = [self.namelist[idx] for idx in range(len(self.namelist)) if idx%AVP != 0]
            labellist = [self.labellist[idx] for idx in range(len(self.labellist)) if idx%AVP != 0]
            self.namelist = namelist
            self.labellist = labellist
        elif phase == 'val':
            namelist = [self.namelist[idx] for idx in range(len(self.namelist)) if idx%AVP == 0]
            labellist = [self.labellist[idx] for idx in range(len(self.labellist)) if idx%AVP == 0]
            self.namelist = namelist
            self.labellist = labellist

        assert len(self.namelist) > 0

        if norm:
            self.trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.testtrans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_name = self.namelist[idx]
        img_root = os.path.join(self.root, img_name)
        img = Image.open(img_root)
        

        if self.phase == 'train' or self.phase == 'val':
            img = self.trans(img)
            img_label = int(self.labellist[idx][1])
            return img, img_label
        else:
            img = self.testtrans(img)
            return img, idx
