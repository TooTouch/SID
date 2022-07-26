import torch
from lib.adversary import cw, deepfool

class DeepFool:
    def __init__(self, model, num_classes, train_mode, step_size):
        self.model = model
        self.n_classes = num_classes
        self.train_mode = train_mode
        self.step_size = step_size

    def __call__(self, inputs, targets):
        _, adv_data = deepfool(self.model, inputs.data.clone(), targets.data.cpu(), n_classes=self.n_classes, train_mode=self.train_mode, step_size=self.step_size)
        adv_data = torch.clamp(adv_data, 0, 1)

        return adv_data.cuda()


class CW:
    def __init__(self, model, weight, loss_str, crop_frac):
        self.model = model
        self.weight = weight
        self.loss_str = loss_str
        self.crop_frac = crop_frac

    def __call__(self, inputs, targets):
        _, adv_data = cw(self.model, inputs.data.clone(), targets.data.cpu(), self.weight, self.loss_str, crop_frac=self.crop_frac)
        adv_data = torch.clamp(adv_data, 0, 1)

        return adv_data
