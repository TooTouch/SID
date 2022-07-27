from torchvision import transforms
from torch.utils.data import DataLoader

import os
import pickle

def create_dataloader(datadir, dataname, batch_size, num_workers):
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

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader