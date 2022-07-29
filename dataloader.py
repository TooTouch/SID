from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import numpy as np
from random import seed, shuffle

def create_dataloader(datadir, dataname, dev_ratio, batch_size, num_workers):
    # Load Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  
    ])

    load_dataset = __import__('torchvision.datasets', fromlist=dataname)

    if dataname == 'SVHN':
        trainset = load_dataset.__dict__[dataname](os.path.join(datadir,dataname), split='train', download=True, transform=transform_train)
        testset = load_dataset.__dict__[dataname](os.path.join(datadir,dataname), split='test', download=True, transform=transform_test)
    else:
        trainset = load_dataset.__dict__[dataname](os.path.join(datadir,dataname), train=True, download=True, transform=transform_train)
        testset = load_dataset.__dict__[dataname](os.path.join(datadir,dataname), train=False, download=True, transform=transform_test)

    
    # split train into train and dev
    num_train = len(trainset)
    indices = list(range(num_train))

    seed(0)
    shuffle(indices)

    split = int(np.floor(dev_ratio * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    devloader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, devloader, testloader