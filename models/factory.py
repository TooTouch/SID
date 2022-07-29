import torch
import torch.nn as nn
import os

class Normalize(nn.Module):
    def __init__(self, model, dataname):
        super(Normalize, self).__init__()

        if dataname.lower() == "cifar10":
            m, s = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        elif dataname.lower() == "cifar100":
            m, s = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        elif dataname.lower() == "svhn":
            m, s = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        self.register_buffer('mean', torch.Tensor(m))
        self.register_buffer('std', torch.Tensor(s))
        
        self.model = model

    def forward(self, inputs):
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        x = (inputs - mean) / std

        return self.model(x)


def create_model(modelname, dataname='CIFAR10', num_classes=10, use_wavelet_transform=False, checkpoint=None, logits_dim=10):
    model = __import__('models').__dict__[modelname](
        num_classes           = num_classes,
        use_wavelet_transform = use_wavelet_transform, 
        logits_dim            = logits_dim
    )

    if modelname != 'detector':
        model = Normalize(model, dataname)

    if checkpoint:
        assert os.path.isfile(checkpoint), "checkpoint does not exist"
        if modelname != 'detector':
            model.model.load_state_dict(torch.load(checkpoint))
        else:
            model.load_state_dict(torch.load(checkpoint))
            

    return model