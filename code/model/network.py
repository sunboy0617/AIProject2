import torch
import torch.nn as nn
import torchvision
from . import block
from pdb import set_trace as st
from . import wide_res_net
from . import wideresidual
from . import resnext
from . import seresnet


class ResNetClassifier(nn.Module):
    def __init__(self, out_nc, pretrained=True):
        super(ResNetClassifier, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, out_nc)

    def forward(self, x):
        return self.model(x) 

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


class SunNetClassifier(nn.Module):
    def __init__(self,out_nc):
        super(SunNetClassifier,self).__init__()
        self.model = block.SunNet(out_nc)

    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

class VGGNetClassifier(nn.Module):
    def __init__(self, out_nc, vggtype):
        super(VGGNetClassifier,self).__init__()
        if vggtype == 11:
            self.model = block.vgg11_bn(out_nc)
        elif vggtype == 13:
            self.model = block.vgg13_bn(out_nc)
        elif vggtype == 16:
            self.model = block.vgg16_bn(out_nc)
        elif vggtype == 19:
            self.model = block.vgg19_bn(out_nc)
        else:
            raise NotImplementedError('VGG type not recognized.')
            
    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


class WideResNetClassifier(nn.Module):
    def __init__(self, out_nc, depth=16, width_factor=8, dropout=0.0):
        super(WideResNetClassifier,self).__init__()
        self.model = wide_res_net.WideResNet(depth, width_factor, dropout, 3, out_nc)
            
    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

class WideResNetClassifier_v2(nn.Module):
    def __init__(self, out_nc, depth, width_factor):
        super(WideResNetClassifier_v2,self).__init__()
        self.model = wideresidual.wideresnet(out_nc, depth, width_factor)
            
    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

class ResNextClassifier(nn.Module):
    def __init__(self, cardinality, out_nc, depth):
        super(ResNextClassifier,self).__init__()
        self.model = resnext.resnext(cardinality=8, depth=depth, num_classes=out_nc, widen_factor=4)
            
    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


class SEResNetClassifier(nn.Module):
    def __init__(self, out_nc):
        super(SEResNetClassifier,self).__init__()
        self.model = seresnet.seresnet152(out_nc)
            
    def forward(self,x):
        return self.model(x)

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            network = self.model.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
