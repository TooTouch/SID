import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class DWTModule(DWTForward):
    def __init__(self, wave='sym17', plug_mode='append'):
        super(DWTModule, self).__init__(J=1, wave=wave, mode='symmetric', Requirs_Grad=True)
        self.plug_mode = plug_mode

    def wavelets(self, x):
        # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',\n
        # 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'\n\n
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        Yl, Yh = self(x)
        output = self.plugdata(x, Yl, Yh, self.plug_mode)
        return output

    def plugdata(self, x, Yl, Yh, mode):
        if mode == 'append':
            output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3]).to(x.device)
            output[:, 0:3, :] = Yl[:, :, :]
            output[:, 3:6, :] = Yh[0][:, 0, :, :]
            output[:, 6:9, :] = Yh[0][:, 1, :, :]
            output[:, 9:12, :] = Yh[0][:, 2, :, :]
            output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        elif mode == 'avg':
            output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3]).to(x.device)
            output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
            output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
            output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
            output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
            output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        return output