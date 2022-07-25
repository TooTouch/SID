"""
from https://github.com/JinyuTian/SID
"""

import torch
import numpy as np


def get_curve(known, novel):
    known.sort()
    novel.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95

def metric(known, novel, verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(known, novel)
    results = dict()

    # TNR
    mtype = 'TNR'
    results[mtype] = tnr_at_tpr95
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
        print('')
    
    return results

def get_auroc(outputs, targets):
    """
    0 (consistency), 1 (inconsistency) -> 0
    2 (adversarial examples) -> 1
    """

    Bioutput = torch.zeros([outputs.shape[0], 2])
    Bioutput[:, 0] = torch.max(outputs[:, 0:2], 1)[0]
    Bioutput[:, 1] = outputs[:, 2]

    targets[np.nonzero(targets.cpu().numpy() == 1)] = 0
    targets[np.nonzero(targets.cpu().numpy() == 2)] = 1

    Bioutput = torch.nn.Softmax(dim=1)(Bioutput)

    y_pred = Bioutput.detach().cpu().numpy().astype(np.float64)[:, 1]
    Y = targets.detach().cpu().numpy().astype(np.float64)

    # TODO: why negative?
    known = -y_pred[np.where(Y==0)]
    novel = -y_pred[np.where(Y==1)]

    results = metric(known, novel)
    return results