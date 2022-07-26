import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

class FDResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,in_C=12,wave='sym17',mode='append'):
        super(FDResNet, self).__init__()
        self.wave = wave
        self.DWT = DWTForward(J=1, wave = self.wave, mode='symmetric',Requirs_Grad=True).cuda()
        self.FDmode = mode
        self.in_planes = 64
        self.conv1 = conv3x3(in_C, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.w = Parameter(torch.Tensor(4))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.wavelets(x, self.FDmode)
        # x = self.w[0]*x[:,0:3,:]+self.w[1]*x[:,3:6,:]+self.w[2]*x[:,6:9,:]+self.w[3]*x[:,9:12,:]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

    def wavelets(self, x, FDmode):
        # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',\n
        # 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'\n\n
        x = x.cuda().reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        Yl, Yh = self.DWT(x)
        output = self.plugdata(x, Yl, Yh, FDmode)
        return output

    def plugdata(self, x, Yl, Yh, mode):
        if mode == 'append':
            output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0:3, :] = Yl[:, :, :]
            output[:, 3:6, :] = Yh[0][:, 0, :, :]
            output[:, 6:9, :] = Yh[0][:, 1, :, :]
            output[:, 9:12, :] = Yh[0][:, 2, :, :]
            output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        elif mode == 'avg':
            output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
            output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
            output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
            output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
            output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def resnet34(num_classes, use_wavelet_transform=False, **kwargs):
    if use_wavelet_transform:
        return FDResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, in_C=12)
    else:
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
        
    