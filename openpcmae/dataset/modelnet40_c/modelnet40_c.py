#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 22:13
# @Author  : wangjie
import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# dataset of jiachen sun
class ModelNet40C(Dataset):
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']
    def __init__(self,
                 data_dir='./data/ModelNet40C/modelnet40_c',
                 num_points=1024,
                 corruption=None,
                 severity=None,
                 **kwargs):
        self.num_points = num_points
        self.corruption = corruption
        self.severity = severity

        # h5_name = os.path.join(
        #     data_dir, f'{split}.h5')
        data_path = os.path.join(data_dir, 'data_' + corruption + '_' +str(severity) + '.npy')
        label_path = os.path.join(data_dir, 'label.npy')

        if not osp.isfile(data_path):
            raise FileExistsError(
                f'{data_path} does not exist, please download dataset at first')
        # with h5py.File(h5_name, 'r') as f:
        #     self.points = np.array(f['data']).astype(np.float32)
        #     self.labels = np.array(f['label']).astype(int)
        self.points = np.load(data_path)
        self.labels = np.load(label_path)
        logging.info(f'Successfully load ModelNet40-C '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')


    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        return current_points, label

    def __len__(self):
        return self.points.shape[0]