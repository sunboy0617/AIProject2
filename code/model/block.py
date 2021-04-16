import torch
import torch.nn as nn
from pdb import set_trace as st


class SunNet(nn.Module):
    def __init__(self,out_nc):
        super(SunNet,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2)) 
        
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3),  
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))  
        
        self.fc = nn.Sequential(
                nn.Linear(128 * 5 * 5,512),
                nn.ReLU(inplace=True),
                nn.Linear(512,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,out_nc))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # st()
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x



cfg = {
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    '19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=20):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg16_bn(out_nc):
    return VGG(make_layers(cfg['16'], batch_norm=True), num_class = out_nc)

def vgg19_bn(out_nc):
    return VGG(make_layers(cfg['19'], batch_norm=True), num_class = out_nc)