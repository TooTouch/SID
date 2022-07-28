"""
1. clean, noise, adv example, and target 불러오기
2. model, model_dwt, and dual classifier 만들기


[get logits]
- clean, noise, and adv example 각각 적용
1. model and model dwt 두 모델 사용
2. model & model dwt 같은 예측 or 다른 예측 두개 logit 나누기
return 
1. both logits: (model x model dwt) x (consistency index) x (num classes)
2. only logits: (model x model dwt) x (inconsistency index) x (num classes)
3. both targets: (consistency targets)
4. only targets: (inconsistency targets)
5. label: (consistency) = 0, (inconsistency) = 1, (adv example) = 2

both logits <-> both targets
only logits <-> only targets


[train detector]
1. use concatenated model logits & model dwt logits as inputs
2. train detector
3. evaluate

"""

import logging
import os
import json
import pickle
import argparse
import pickle
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from log import setup_default_logging
from models import create_model
from utils import torch_seed, AverageMeter
from metrics import get_auroc

_logger = logging.getLogger('known attack')



def get_logits(model, model_dwt, images, targets, batch_size, train_ratio, seed, adv_examples=False, device='cpu'):
    
    # init
    consistency_logits = torch.Tensor([])
    inconsistency_logits = torch.Tensor([])

    consistency_targets = torch.Tensor([])
    inconsistency_targets = torch.Tensor([])

    dataloader = DataLoader(
        TensorDataset(images, targets),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1
    )

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            # outputs
            outputs = model(inputs).cpu()
            outputs_dwt = model_dwt(inputs).cpu()

            # preds
            preds = outputs.max(1)[1]
            preds_dwt = outputs_dwt.max(1)[1]

            # consistency
            consistency_idx = torch.nonzero((preds - preds_dwt)==0).squeeze(1)
        
            consistency_logits = torch.cat([
                consistency_logits,
                torch.stack([outputs[consistency_idx], outputs_dwt[consistency_idx]], dim=1)
            ], dim=0)
            
            consistency_targets = torch.cat([
                consistency_targets,
                targets[consistency_idx]
            ])


            # inconsistency
            if not adv_examples:
                inconsistency_idx = torch.nonzero((preds - preds_dwt)!=0).squeeze(1)

                inconsistency_logits = torch.cat([
                    inconsistency_logits,
                    torch.stack([outputs[inconsistency_idx], outputs_dwt[inconsistency_idx]], dim=1)
                ], dim=0)

                inconsistency_targets = torch.cat([
                    inconsistency_targets,
                    targets[inconsistency_idx]
                ])
    

    # train and test split
    consistency_train_size = int(len(consistency_targets) * train_ratio)
    inconsistency_train_size = int(len(inconsistency_targets) * train_ratio)

    torch_seed(seed)
    consistency_random_idx = np.random.permutation(len(consistency_targets))
    inconsistency_random_idx = np.random.permutation(len(inconsistency_targets))


    # train and test labels
    if not adv_examples:
        consistency_labels = torch.zeros_like(consistency_targets).long()
        inconsistency_labels = torch.ones_like(inconsistency_targets).long()

        train_labels = torch.cat([
            consistency_labels[consistency_random_idx[:consistency_train_size]],
            inconsistency_labels[inconsistency_random_idx[:inconsistency_train_size]]
        ])

        test_labels = torch.cat([
            consistency_labels[consistency_random_idx[consistency_train_size:]],
            inconsistency_labels[inconsistency_random_idx[inconsistency_train_size:]]
        ])
    else:
        labels = 2 * torch.ones_like(consistency_targets).long()
        train_labels = labels[consistency_random_idx[:consistency_train_size]]
        test_labels = labels[consistency_random_idx[consistency_train_size:]]


    # train and test logits
    train_logits =  {
        'consistency':{
            'logits' :consistency_logits[consistency_random_idx[:consistency_train_size]], 
            'targets':consistency_targets[consistency_random_idx[:consistency_train_size]]
        },
        'labels':train_labels
    }

    test_logits =  {
        'consistency':{
            'logits' :consistency_logits[consistency_random_idx[consistency_train_size:]], 
            'targets':consistency_targets[consistency_random_idx[consistency_train_size:]]
        },
        'labels':test_labels
    }

    if not adv_examples:
        train_logits.update({
            'inconsistency':{
                'logits' :inconsistency_logits[inconsistency_random_idx[:inconsistency_train_size]], 
                'targets':inconsistency_targets[inconsistency_random_idx[:inconsistency_train_size]]
            }
        })
        
        test_logits.update({
            'inconsistency':{
                'logits' :inconsistency_logits[inconsistency_random_idx[inconsistency_train_size:]], 
                'targets':inconsistency_targets[inconsistency_random_idx[inconsistency_train_size:]]
            }
        })

    return train_logits, test_logits


def get_stack_logits(model, model_dwt, save_path, batch_size, train_ratio, seed, savedir, device='cpu'):

    # if os.os.path.join(savedir, 'train.pt')
    bucket = pickle.load(open(save_path, 'rb'))

    clean_train_logits, clean_test_logits = get_logits(
        model        = model, 
        model_dwt    = model_dwt, 
        images       = bucket['clean'], 
        targets      = bucket['targets'], 
        batch_size   = batch_size, 
        train_ratio  = train_ratio,
        seed         = seed,
        adv_examples = False, 
        device       = device
    )

    noise_train_logits, noise_test_logits = get_logits(
        model        = model, 
        model_dwt    = model_dwt, 
        images       = bucket['noise'], 
        targets      = bucket['targets'], 
        batch_size   = batch_size, 
        train_ratio  = train_ratio,
        seed         = seed,
        adv_examples = False, 
        device       = device
    )

    adv_train_logits, adv_test_logits = get_logits(
        model        = model, 
        model_dwt    = model_dwt, 
        images       = bucket['adv'], 
        targets      = bucket['targets'], 
        batch_size   = batch_size, 
        train_ratio  = train_ratio,
        seed         = seed,
        adv_examples = True, 
        device       = device
    )

    train_logits, train_labels, test_logits, test_labels = save_logits(
        clean_logits = [clean_train_logits, clean_test_logits],
        noise_logits = [noise_train_logits, noise_test_logits],
        adv_logits   = [adv_train_logits, adv_test_logits],
        savedir      = savedir
    )

    return train_logits, train_labels, test_logits, test_labels


def save_logits(clean_logits: list, noise_logits: list, adv_logits: list, savedir: str):
    clean_train_logits, clean_test_logits = clean_logits
    noise_train_logits, noise_test_logits = noise_logits
    adv_train_logits, adv_test_logits = adv_logits

    # logits
    train_logits = torch.cat([
        clean_train_logits['consistency']['logits'],
        clean_train_logits['inconsistency']['logits'],
        noise_train_logits['consistency']['logits'],
        noise_train_logits['inconsistency']['logits'],
        adv_train_logits['consistency']['logits']
    ], dim=0)

    test_logits = torch.cat([
        clean_test_logits['consistency']['logits'],
        clean_test_logits['inconsistency']['logits'],
        noise_test_logits['consistency']['logits'],
        noise_test_logits['inconsistency']['logits'],
        adv_test_logits['consistency']['logits']
    ], dim=0)

    # labels
    train_labels = torch.cat([
        clean_train_logits['labels'],
        noise_train_logits['labels'],
        adv_train_logits['labels']
    ], dim=0)

    test_labels = torch.cat([
        clean_test_logits['labels'],
        noise_test_logits['labels'],
        adv_test_logits['labels']
    ], dim=0)

    # save
    save_dict = {
        'train':{
            'logits':train_logits, 
            'labels':train_labels, 
        },
        'test':{
            'logits':test_logits, 
            'labels':test_labels,
        },
        'clean':{
            'train':clean_train_logits,
            'test':clean_test_logits
        }
    }
    for k, v in save_dict.items():
        torch.save(v, os.path.join(savedir, f'{k}.pt'))

    return train_logits, train_labels, test_logits, test_labels


def train(model, dataloader, criterion, optimizer, log_interval=-1, verbose=True, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    outputs_total = torch.Tensor([])
    targets_total = torch.Tensor([])

    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs[:,0,:], inputs[:,1,:])
        loss = criterion(outputs, targets)

        loss.backward()
        
        # loss update
        optimizer.step()
        optimizer.zero_grad()

        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))

        # stack outputs and targets
        outputs_total = torch.cat([outputs_total, outputs.detach().cpu()], dim=0)
        targets_total = torch.cat([targets_total, targets.detach().cpu()], dim=0)

        batch_time_m.update(time.time() - end)
    
        if (idx % log_interval == 0 and idx != 0) and verbose: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                         'Acc: {acc.avg:.3%} '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                         'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                         idx+1, len(dataloader), 
                         loss       = losses_m, 
                         acc        = acc_m, 
                         lr         = optimizer.param_groups[0]['lr'],
                         batch_time = batch_time_m,
                         rate       = inputs.size(0) / batch_time_m.val,
                         rate_avg   = inputs.size(0) / batch_time_m.avg,
                         data_time  = data_time_m))
   
        end = time.time()

    # auroc
    results = get_auroc(outputs_total, targets_total)
    auroc = results['AUROC']

    _logger.info('TRAIN [FINAL] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                 'AUROC: {auroc:.4f}  '
                 'Acc: {acc.avg:.3%}'.format(
                 loss       = losses_m, 
                 auroc      = auroc,
                 acc        = acc_m))

        
def test(model, dataloader, criterion, log_interval=-1, verbose=True, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    
    outputs_total = torch.Tensor([])
    targets_total = torch.Tensor([])

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs[:,0,:], inputs[:,1,:])
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            # stack outputs and targets
            outputs_total = torch.cat([outputs_total, outputs.detach().cpu()], dim=0)
            targets_total = torch.cat([targets_total, targets.detach().cpu()], dim=0)

            if (idx % log_interval == 0 and idx != 0) and verbose: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

    # auroc
    results = get_auroc(outputs_total, targets_total)
    auroc = results['AUROC']
    _logger.info('TEST [FIENAL]: Loss: %.3f | AUROC: %.3f | Acc: %.3f%% [%d/%d]' % 
                (total_loss/(idx+1), auroc, 100.*correct/total, correct, total))
        
    return results


def train_detector(
    model, model_dwt, detector,
    save_bucket_path, savedir, 
    epochs, batch_size, train_ratio, 
    log_interval, seed, device='cpu'
):

    # get logits
    train_logits, train_labels, test_logits, test_labels = get_stack_logits(
        model       = model, 
        model_dwt   = model_dwt, 
        save_path   = save_bucket_path, 
        batch_size  = batch_size, 
        train_ratio = train_ratio, 
        seed        = seed, 
        savedir     = savedir,
        device      = device
    )

    # make dataloader
    trainloader = DataLoader(
        TensorDataset(train_logits, train_labels),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 1
    )

    testloader = DataLoader(
        TensorDataset(test_logits, test_labels),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1
    )

    # setting
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        detector.parameters(), 
        lr           = 0.005,
        momentum     = 0.8,
        weight_decay = 5e-3
    )

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40,50,60,70], gamma=1.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train and test
    best_auroc = 0
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        train(detector, trainloader, criterion, optimizer, log_interval, True, device)
        results = test(detector, testloader, criterion, log_interval, True, device)

        auroc = results['AUROC']
        if auroc > best_auroc:
            best_auroc = auroc

            state = {'best_epoch':epoch}
            state.update(results)
            json.dump(state, open(os.path.join(savedir, 'result.json'), 'w'), indent=4)
            torch.save(detector.state_dict(), os.path.join(savedir, 'detector.pt'))

        scheduler.step()


def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    savedir = os.path.join(args.savedir,args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # save argsvars
    json.dump(vars(args), open(os.path.join(savedir, 'args.json'), 'w'), indent=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))
    
    # Build Model
    model = create_model(
        modelname             = args.modelname, 
        dataname              = args.dataname,
        num_classes           = args.num_classes, 
        use_wavelet_transform = False,
        checkpoint            = args.model_checkpoint
    ).to(device)
    model.eval()

    model_dwt = create_model(
        modelname             = args.modelname, 
        dataname              = args.dataname,
        num_classes           = args.num_classes, 
        use_wavelet_transform = True,
        checkpoint            = args.model_dwt_checkpoint
    ).to(device)
    model_dwt.eval()
    
    detector = create_model(
        modelname   = 'detector',
        num_classes = 3,
        logits_dim  = args.num_classes
    ).to(device)

    # validate
    train_detector(
        model            = model, 
        model_dwt        = model_dwt, 
        detector         = detector,
        save_bucket_path = args.save_bucket_path, 
        savedir          = savedir, 
        epochs           = args.epochs, 
        batch_size       = args.batch_size, 
        train_ratio      = args.train_ratio, 
        log_interval     = args.log_interval, 
        seed             = args.seed, 
        device           = device
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--modelname',type=str,choices=['vgg19','resnet34'])

    # checkpoint
    parser.add_argument('--model_checkpoint',type=str,help='model checkpoint path')
    parser.add_argument('--model_dwt_checkpoint',type=str,help='dwt model checkpoint path')

    # dataset
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--save_bucket_path',type=str,help='saved bucket path')
    parser.add_argument('--dataname',type=str,default='CIFAR10',choices=['CIFAR10','CIFAR100','SVHN'],help='data name')
    parser.add_argument('--num_classes',type=int,default=10,help='the number of classes')

    # training
    parser.add_argument('--epochs',type=int,default=100,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--train_ratio',type=float,default=0.6,help='train ratio')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    args = parser.parse_args()

    run(args)