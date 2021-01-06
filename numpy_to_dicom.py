#change validation and train paths
#change the path to the label csvs in load_cases
#change directory to write dicoms out to

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm_notebook

valid_path = '/home/niamh/Documents/MRNET/data/valid'
train_path = '/home/niamh/Documents/MRNET/data/train'


def load_one_stack(case, data_path=valid_path, plane='coronal'):
    fpath = '{}/{}/{}.npy'.format(data_path, plane, case)
    return np.load(fpath)

def load_stacks(case, data_path=valid_path):
    x = {}
    planes = ['coronal', 'sagittal', 'axial']
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, data_path, plane=plane)
    return x

def load_cases(train=True, n=None):
    assert (type(n) == int) and (n < 1250)
    if train:
        case_list = pd.read_csv('/home/niamh/Documents/MRNET/data/train-acl.csv', names=['case', 'label'], header=None,
                               dtype={'case': str, 'label': np.int64})['case'].tolist()
    else:
        case_list = pd.read_csv('/home/niamh/Documents/MRNET/data/valid-acl.csv', names=['case', 'label'], header=None,
                               dtype={'case': str, 'label': np.int64})['case'].tolist()
    cases = {}
    if train:
        data_path=train_path
    else:
        data_path= valid_path

    if n is not None:
        case_list = case_list[:n]

    for case in tqdm_notebook(case_list, leave=False):
        x = load_stacks(case, data_path)
        cases[case] = x
    return cases

#create dictionary cases
cases = load_cases(train=True, n=1130)


for i in range (0,1300):
    print(i)
    if len(str(i)) == 1:
        string = '000'+str(i)
    elif len(str(i))==2:
        string = '00'+str(i)
    else:
        string = '0'+str(i)
    img = sitk.GetImageFromArray(cases[string]['sagittal'])
    img.SetMetaData('0008|0060', 'MR')
    s = string + '_' + 'sagittal'
    img.SetMetaData('0010|0010', s)
    #write out dicom
    sitk.WriteImage(img, "/home/niamh/Documents/MRNET/data-dicom/train/sagittal/"+string + ".dcm")
