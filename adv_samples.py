"""
successed adversarial examples

1. test loader 불러오기
2. pretrained model & DWT model 불러오기


1. clean, noise, adv example 생성
2. clean & noise 정분류 - adv example 오분류 이미지에 대해 attack success 이미지에 대한 index 저장
3. attack success 케이스의 clean image, noise image, adv example, label 4개 저장
"""

import logging
import os
import json
import torch
import pickle
import argparse

from dataloader import create_dataloader
from log import setup_default_logging
from models import create_model
from utils import torch_seed, AverageMeter, extract_correct
from adv_attacks import adv_attack


_logger = logging.getLogger('adv sample')


def make_bucket():
    successed_images = {}
    for k in ['clean','noise','adv','targets']:
        successed_images[k] = torch.Tensor([])    

    return successed_images


def extract_success(
    bucket, 
    inputs, inputs_noise, inputs_adv, targets,
    correct_clean, correct_noise, correct_adv
):
    # check success
    for i in range(inputs.size(0)):
        if correct_clean[i] and correct_noise[i] and not correct_adv[i]:
            bucket['clean'] = torch.cat([bucket['clean'], inputs[[i]].detach().cpu()])
            bucket['noise'] = torch.cat([bucket['noise'], inputs_noise[[i]].detach().cpu()])
            bucket['adv'] = torch.cat([bucket['adv'], inputs_adv[[i]].detach().cpu()])
            bucket['targets'] = torch.cat([bucket['targets'], targets[[i]].detach().cpu()])

    return bucket

def validate(model, model_dwt, testloader, adv_method, adv_params, noise_size, savedir, log_interval=1, device='cpu'):

    clean_acc = AverageMeter()
    noise_acc = AverageMeter()
    adv_acc = AverageMeter()
    clean_dwt_acc = AverageMeter()
    adv_dwt_acc = AverageMeter()

    successed_images = make_bucket()
    
    atk = adv_attack(model, adv_method, adv_params)

    model.eval()
    model_dwt.eval()
    for i, (inputs, targets) in enumerate(testloader):

        # clean, noise, and adv examples
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_noise = torch.add(inputs, torch.randn_like(inputs).to(device), alpha=noise_size)
        inputs_adv = atk(inputs, targets)

        # clean pred
        outputs = model(inputs)
        correct_clean = extract_correct(outputs, targets)

        outputs_dwt = model_dwt(inputs)
        correct_dwt_clean = extract_correct(outputs_dwt, targets)

        # noise pred
        outputs_noise = model(inputs_noise)
        correct_noise = extract_correct(outputs_noise, targets)
        
        # adv pred
        outputs_adv = model(inputs_adv)
        correct_adv = extract_correct(outputs_adv, targets)

        outputs_dwt_adv = model_dwt(inputs_adv)
        correct_dwt_adv = extract_correct(outputs_dwt_adv, targets)

        # check success
        successed_images = extract_success(
            bucket        = successed_images,
            inputs        = inputs,
            inputs_noise  = inputs_noise,
            inputs_adv    = inputs_adv,
            targets       = targets,
            correct_clean = correct_clean,
            correct_noise = correct_noise,
            correct_adv   = correct_adv
        )
        
        # accuracy
        clean_acc.update(correct_clean.sum().item()/targets.size(0), n=targets.size(0))
        noise_acc.update(correct_noise.sum().item()/targets.size(0), n=targets.size(0))
        adv_acc.update(correct_adv.sum().item()/targets.size(0), n=targets.size(0))
        clean_dwt_acc.update(correct_dwt_clean.sum().item()/targets.size(0), n=targets.size(0))
        adv_dwt_acc.update(correct_dwt_adv.sum().item()/targets.size(0), n=targets.size(0))
        

        if i % log_interval == 0 and i != 0: 
            _logger.info('TEST [{:>4d}/{}] '
                         'CLEAN: {clean.val:>6.4f} ({clean.avg:>6.4f}) '
                         'CLEAN DWT: {clean_dwt.val:>6.4f} ({clean_dwt.avg:>6.4f}) '
                         'ADV: {adv.val:>6.4f} ({adv.avg:>6.4f}) '
                         'ADV DWT: {adv_dwt.val:>6.4f} ({adv_dwt.avg:>6.4f}) '
                         'NOISE: {noise.val:>6.4f} ({noise.avg:>6.4f})'.format(
                             i+1, len(testloader),
                             clean     = clean_acc,
                             clean_dwt = clean_dwt_acc,
                             adv       = adv_acc,
                             adv_dwt   = adv_dwt_acc,
                             noise     = noise_acc
                         ))
                    

    _logger.info('TEST [FINAL] '
                 'CLEAN: {clean.avg:>6.4f} '
                 'CLEAN DWT: {clean_dwt.avg:>6.4f} '
                 'ADV: {adv.avg:>6.4f} '
                 'ADV DWT: {adv_dwt.avg:>6.4f} '
                 'NOISE: {noise.avg:>6.4f}'.format(
                     i+1, len(testloader),
                     clean     = clean_acc,
                     clean_dwt = clean_dwt_acc,
                     adv       = adv_acc,
                     adv_dwt   = adv_dwt_acc,
                     noise     = noise_acc
                 ))
    

    # save successed images
    pickle.dump(successed_images, open(os.path.join(savedir, 'successed_images.pkl'),'wb'))

    # save results
    json.dump(
        {
            'clean acc':clean_acc.avg,
            'noise acc':noise_acc.avg,
            'adv acc':adv_acc.avg,
            'clean dwt acc':clean_dwt_acc.avg,
            'adv dwt acc':adv_dwt_acc.avg
        }, 
        open(os.path.join(savedir, 'results.json'),'w'),
        indent=4
    )


def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    savedir = os.path.join(args.savedir,args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # load adversarial parameteres and update arguments
    adv_params = json.load(open(os.path.join(args.adv_config, f'{args.adv_name.lower()}.json'),'r'))
    vars(args).update(adv_params)

    # save argsvars
    json.dump(vars(args), open(os.path.join(savedir, 'args.json'), 'w'), indent=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    _, testloader = create_dataloader(
        datadir     = args.datadir, 
        dataname    = args.dataname, 
        batch_size  = args.batch_size, 
        num_workers = args.num_workers
    )
    
    # Build Model
    model = create_model(
        modelname             = args.modelname, 
        num_classes           = args.num_classes, 
        use_wavelet_transform = False,
        checkpoint            = args.model_checkpoint
    ).to(device)

    model_dwt = create_model(
        modelname             = args.modelname, 
        num_classes           = args.num_classes, 
        use_wavelet_transform = True,
        checkpoint            = args.model_dwt_checkpoint
    ).to(device)
    

    # validate
    validate(
        model        = model, 
        model_dwt    = model_dwt, 
        testloader   = testloader, 
        adv_method   = args.adv_method, 
        adv_params   = adv_params, 
        noise_size   = args.noise_size, 
        savedir      = savedir, 
        log_interval = args.log_interval, 
        device       = device
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--modelname',type=str,choices=['vgg19','resnet34'])
    parser.add_argument('--adv_name',type=str,help='adversrial experiments name')

    # checkpoint
    parser.add_argument('--model_checkpoint',type=str,help='model checkpoint path')
    parser.add_argument('--model_dwt_checkpoint',type=str,help='dwt model checkpoint path')

    # adv
    parser.add_argument('--adv_method',type=str,help='adversarial attack method name')
    parser.add_argument('--adv_config',type=str,default='./configs_adv',help='adversarial attack configuration directory')
    parser.add_argument('--noise_size',type=float,default=0.01,help='noise size')

    # dataset
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--dataname',type=str,default='CIFAR10',choices=['CIFAR10','CIFAR100','SVHN'],help='data name')
    parser.add_argument('--num_classes',type=int,default=10,help='the number of classes')

    # training
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--num-workers',type=int,default=8,help='the number of workers (threads)')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    args = parser.parse_args()

    run(args)