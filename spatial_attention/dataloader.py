import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd

import pdb
#MEAN = 58.81274059207973
#STDDEV = 48.56406668573295
from torch.autograd import Variable

class MRDataset(data.Dataset):
    def __init__(self, indexes, root_dir, task, plane, valid=False, transform=None, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.valid=valid
        if self.valid == True:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)

            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
            self.records = self.records.iloc[indexes,:].reset_index(drop=True)
        
        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = [1, neg / pos]
        else:
            self.weights = weights

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        label = torch.FloatTensor([label])

        if self.transform:
            array = self.transform(array)
            array = array.numpy()
            
        #else:
          #  array = np.stack((array,)*3, axis=1)
          #  array = torch.FloatTensor(array)
        
      #  array2=np.insert(array[0:array.shape[0]-1], 0, array[0], axis=0)
       # array3=np.insert(array[0:array.shape[0]-2], 0, array[0], axis=0)
    #    array3=np.insert(array3, 0, array[0], axis=0)
     #   array = np.stack((array,array2,array3))
        array = np.stack((array,)*3, axis=1)
        array = torch.FloatTensor(array)
     #   array=array.permute(1,0,2,3)
        
        #array = (array - MEAN) / STDDEV


        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return array, label, weight


