"""
1. load adv samples path list
2. load detector
3. eval adv examples by adv methods
4. save 
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import argparse
from glob import glob

import torch
from torch.utils.data import TensorDataset, DataLoader

from log import setup_default_logging
from utils import torch_seed
from models import create_model
from known_attack import test


_logger = logging.getLogger('transfer attack')



def transfer(target_detector_path, source_adv_test_path, logits_dim, batch_size, device='cpu'):
    # load detector
    detector = create_model(
        modelname   = 'detector',
        num_classes = 3,
        logits_dim  = logits_dim,
        checkpoint  = target_detector_path
    ).to(device)

    # load dataset
    testset = torch.load(source_adv_test_path)

    # make dataloader
    testloader = DataLoader(
        TensorDataset(testset['logits'], testset['labels']),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1
    )

    # setting
    criterion = torch.nn.CrossEntropyLoss()

    results = test(
        model        = detector, 
        dataloader   = testloader, 
        criterion    = criterion, 
        verbose      = False,
        device       = device
    )

    return results

def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # save argsvars
    json.dump(vars(args), open(os.path.join(savedir, 'args.json'), 'w'), indent=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # transfer attack
    adv_list = glob(os.path.join(args.known_attack_path, '*'))

    attack_results = pd.DataFrame(
        np.zeros((len(adv_list), len(adv_list))),
        columns = [os.path.basename(p) for p in adv_list],
        index   = [os.path.basename(p) for p in adv_list]
    )

    for target_adv_path in adv_list:
        target_adv = os.path.basename(target_adv_path)

        for source_adv_path in adv_list:
            source_adv = os.path.basename(source_adv_path)

            _logger.info('Target: {} | Source: {}'.format(target_adv, source_adv))
            results = transfer(
                target_detector_path = os.path.join(target_adv_path, 'detector.pt'),
                source_adv_test_path = os.path.join(source_adv_path, 'test.pt'),
                logits_dim           = args.num_classes,
                batch_size           = args.batch_size,
                device               = device
            )

            attack_results.loc[source_adv, target_adv] = results['AUROC'] * 100

    # save
    attack_results.round(2).to_csv(os.path.join(savedir, 'transfer_results.csv'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--known_attack_path',type=str,help='known attack path')

    # dataset
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--dataname',type=str,default='CIFAR10',choices=['CIFAR10','CIFAR100','SVHN'],help='data name')
    parser.add_argument('--num_classes',type=int,default=10,help='the number of classes')

    # training
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    args = parser.parse_args()

    run(args)