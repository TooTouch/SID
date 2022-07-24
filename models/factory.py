import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor((0.4914, 0.4822, 0.4465)))
        self.register_buffer('std', torch.Tensor((0.2023, 0.1994, 0.2010)))
        
    def forward(self, input):
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return (input - mean) / std


def create_model(modelname, num_classes=10, use_wavelet_transform=False, checkpoint=None, logits_dim=10):
    model = __import__('models').__dict__[modelname](
        num_classes           = num_classes,
        use_wavelet_transform = use_wavelet_transform, 
        logits_dim            = logits_dim
    )

    if modelname != 'detector':
        model = nn.Sequential(
            Normalize(),
            model
        )

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    return model