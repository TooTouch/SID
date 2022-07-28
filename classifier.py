import numpy as np
import os
import time
import json
import wandb
import logging
from collections import OrderedDict

import torch
import argparse

from log import setup_default_logging
from models import create_model
from dataloader import create_dataloader
from utils import torch_seed, AverageMeter

_logger = logging.getLogger('train')


def train(model, dataloader, criterion, optimizer, log_interval, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        
        # loss update
        optimizer.step()
        optimizer.zero_grad()

        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
    
        if idx % log_interval == 0 and idx != 0: 
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
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def fit(
    exp_name, model, epochs, trainloader, testloader, criterion, optimizer, scheduler, 
    savedir, log_interval, use_wandb, device='cpu'
):
    
    save_model_path = os.path.join(savedir, f'{exp_name}.pt')

    if not os.path.isfile(save_model_path):

        best_acc = 0

        for epoch in range(epochs):
            _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
            train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device)
            eval_metrics = test(model, testloader, criterion, log_interval, device)

            scheduler.step()

            # wandb
            if use_wandb:
                metrics = OrderedDict()
                metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
                metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
                wandb.log(metrics)
            
            # checkpoint
            if best_acc < eval_metrics['acc']:
                state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
                json.dump(state, open(os.path.join(savedir, f'{exp_name}.json'),'w'), indent=4)

                torch.save(model.model.state_dict(), save_model_path)
                
                _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

                best_acc = eval_metrics['acc']


        _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))
    else:
        eval_metrics = test(model, testloader, criterion, log_interval, device)

        state = {'best_acc':eval_metrics['acc']}
        json.dump(state, open(os.path.join(savedir, f'{exp_name}_check.json'),'w'), indent=4)


def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # save args
    json.dump(vars(args), open(os.path.join(savedir, 'args.json'), 'w'), indent=4)

    if args.use_wandb:
        wandb.init(name=args.exp_name, project='SID classfier', config=args)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))


    trainloader, testloader = create_dataloader(
        datadir     = args.datadir, 
        dataname    = args.dataname, 
        batch_size  = args.batch_size, 
        num_workers = args.num_workers,
    )
    
    # Build Model
    model = create_model(
        modelname             = args.modelname, 
        dataname              = args.dataname,
        num_classes           = args.num_classes, 
        use_wavelet_transform = args.use_wavelet_transform,
        checkpoint            = args.checkpoint
    )
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # Set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler
    if args.modelname == 'resnet34':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,250], gamma=0.1)
    elif args.modelname == 'vgg19':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1)

    # Fitting model
    fit(exp_name     = args.exp_name,
        model        = model, 
        epochs       = args.epochs, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        savedir      = savedir,
        log_interval = args.log_interval,
        device       = device,
        use_wandb    = args.use_wandb
    )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--modelname',type=str,choices=['vgg19','resnet34'])
    parser.add_argument('--checkpoint',type=str,help='model checkpoint')

    # dataset
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--dataname',type=str,default='CIFAR10',choices=['CIFAR10','CIFAR100','SVHN'],help='data name')
    parser.add_argument('--num_classes',type=int,default=10,help='the number of classes')

    # training
    parser.add_argument('--epochs',type=int,default=300,help='the number of epochs')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--num-workers',type=int,default=8,help='the number of workers (threads)')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    parser.add_argument('--use_wavelet_transform',action='store_true',help='use discrete wavelet trasnform')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')

    args = parser.parse_args()

    run(args)