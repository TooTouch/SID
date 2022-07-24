import torch.nn as nn
import torch

class Detector(nn.Module):
    '''
    ModelMnist model
    '''
    def __init__(self, num_classes=3, logits_dim=10):
        super(Detector, self).__init__()
        
        self.clf = nn.Sequential(
            nn.Linear(2*logits_dim, 2*logits_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*logits_dim, num_classes)
        )
        
    def forward(self, logits, logits_dwt):
        output = torch.cat((logits, logits_dwt),1)
        output = self.clf(output)
        return output


def detector(num_classes, logits_dim, **kwargs):
    return Detector(num_classes, logits_dim)