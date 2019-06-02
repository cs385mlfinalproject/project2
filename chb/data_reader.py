
# coding: utf-8

# In[1]:


import torch 
import numpy as np
import cv2
import pickle
from torchvision import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# In[2]:


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])


# In[3]:


def get_data_array(dataset_name, path):
    """ 
    read dataset as numpy array, for different dataset has different 
    storage format, you need to implement it specifically!
    input args: dataset_name, path of the folder
    output: train_X, train_y, test_X, test_y
    X shape: n_samples * H * W * C (c is channels)
    Y shape: n_samples
    """
    train_X, train_y, test_X, test_y = None, None, None, None
    if dataset_name.lower() == 'cifar-10':
        for i in range(1, 6):
            with open("{}/data_batch_{}".format(path, i), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            if train_X is None:
                train_X = dic[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
                train_y = np.array(dic[b'labels'])
            else:
                train_X = np.concatenate((train_X, dic[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)), axis=0)
                train_y = np.concatenate((train_y, np.array(dic[b'labels'])), axis=0)
            
        with open("{}/test_batch".format(path), 'rb') as f:
            dic = pickle.load(f, encoding='bytes')
        test_X = dic[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        test_y = np.array(dic[b'labels'])
    # add your code like : 
    # elif dataset_name.lower() == name:
    
    else:
        raise ValueError
    return train_X.astype('float'), train_y.astype('long'), test_X.astype('float'), test_y.astype('long')


# In[4]:


# train_X, train_y, test_X, test_y = get_data_array('cifar-10', 'data/cifar-10-batches-py')
# print(train_X.shape, train_y.shape)
# print(test_X.shape, test_y.shape)
# print(train_y.dtype)


# In[5]:


class MyDataset(Dataset):

    def __init__(self, X, y, transform=None):
        
        self.X = X
        self.y = y
        self.transform = transform
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        img = cv2.resize(img, (224, 224))
        if self.transform:
            transformed_X = self.transform(img)

        return transformed_X, label


# In[6]:


# dataset_train = MyDataset(X=train_X, y=train_y, transform=transform)


# In[7]:


# dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)


# In[8]:


def data_loader():
    train_X, train_y, test_X, test_y = get_data_array('cifar-10', 'data/cifar-10-batches-py')
    dataset_train = MyDataset(X=train_X, y=train_y, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
    dataset_test = MyDataset(X=test_X, y=test_y, transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=4)
    return dataloader_train, dataloader_test
    



