import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadirs, diagnosis, transform, use_gpu):
        super().__init__()
        self.transform  = transform
        self.use_gpu = use_gpu

        label_dict = {}
        self.paths = []

        for i, line in enumerate(open('/home/niamh/Documents/MRNET/External_data/data/metadata.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            path = line[10]
            label = line[2]
            label_dict[path] = int(int(label) > diagnosis)

        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir+'/'+file)

        self.labels = [label_dict[path[53:]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        vol = vol[:,pad:-pad,pad:-pad]

        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV

        #if self.transform is not None:
        #    vol = self.transform(vol)
        #    vol=np.array(vol)

        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(diagnosis, use_gpu=False):
    train_dirs = ['/home/niamh/Documents/MRNET/External_data/data/vol08','/home/niamh/Documents/MRNET/External_data/data/vol04','/home/niamh/Documents/MRNET/External_data/data/vol03','/home/niamh/Documents/MRNET/External_data/data/vol09','/home/niamh/Documents/MRNET/External_data/data/vol06','/home/niamh/Documents/MRNET/External_data/data/vol07']
    valid_dirs = ['/home/niamh/Documents/MRNET/External_data/data/vol10','/home/niamh/Documents/MRNET/External_data/data/vol05']
    test_dirs = ['/home/niamh/Documents/MRNET/External_data/data/vol01','/home/niamh/Documents/MRNET/External_data/data/vol02']

    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
       # transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])


    train_dataset = Dataset(train_dirs, diagnosis, augmentor, use_gpu)
    valid_dataset = Dataset(valid_dirs, diagnosis, None, use_gpu)
    test_dataset = Dataset(test_dirs, diagnosis, None, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader
