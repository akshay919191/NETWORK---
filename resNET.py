import torch 
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import transforms

class basicMODULE(nn.Module):
    def __init__(self , inchannel , outchannel , stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannel , outchannel , stride = stride , padding = 1 , bias = False)
        self.b1 = nn.BatchNorm2d(outchannel)

        self.conv2 = nn.Conv2d(outchannel , outchannel , stride = 1 , padding = 1 , bias = False)
        self.b2 = nn.BatchNorm2d(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel , outchannel , kernel_size = 1 , stride = stride),
                                          nn.BatchNorm2d(outchannel)
                                          )
    def forward(self , x):
        residue = x

        x = self.conv1(x)
        x = self.b1(x)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.b2(x)
        
        x += self.shortcut(residue)
        x = F.relu(x)

        return x
    
class ResNET(nn.Module):
    def __init__(self , inchannel , numclass):
        super().__init__()

        # intial layer 
        self.conv1 = nn.Conv2d(inchannel , 64 , kernel_size = 7 , stride = 2 , padding = 1)
        self.b1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 1 , stride = 2 , padding = 1)

        # res-layer
        self.layer1 = self.make_layer(64 , 64 , 2 , stride = 1)
        self.layer2 = self.make_layer(64 , 64 , 2 , stride = 1)
        self.layer3 = self.make_layer(64 , 128 , 2 , stride = 2)
        self.layer4 = self.make_layer(128 , 128 , 2 , stride = 2)
        self.layer5 = self.make_layer(128 , 256 , 2 , stride = 2)
        self.layer6 = self.make_layer(256 , 256 , 2 , stride = 2)
        self.layer7 = self.make_layer(256 , 512 , 2 , stride = 2)
        self.layer8 = self.make_layer(512 , 512 , 2 , stride = 2)
        self.layer9 = self.make_layer(512 , 512 , 2 , stride = 2)

        self.pool = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(512*1 , numclass)

    def make_layer(self , inchannel , outchannel , numchannel , stride = 1):
        layer = []

        layer.append(basicMODULE(inchannel , outchannel, stride))

        for _ in range(1 , numchannel):
            layer.append(basicMODULE(outchannel , outchannel , stride = 1))

    
        return nn.Sequential(*layer)
    
    def forward(self , x):
        out = self.conv1(x)
        out = self.b1(out)
        out = self.act(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out