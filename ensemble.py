import torch.nn as nn
from torchvision.models import resnet101
import torch
from densenet import densenet169
from resnet import resnet101
from VGG import vgg16_bn

# Ensemble of 3 models together
class Ensemble(nn.Module):
    def __init__(self, densenet_path,resnet_path,vgg_path):
        super(Ensemble, self).__init__()

        self.densenet = densenet169(pretrained=True, droprate= 0)
        self.densenet.load_state_dict(torch.load(densenet_path))

        self.resnet = resnet101()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.resnet.load_state_dict(torch.load(resnet_path))

        self.vgg = vgg16_bn()
        self.vgg.classifier[6] = nn.Linear(4096, 1)
        self.vgg.load_state_dict(torch.load(vgg_path))

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        return x1,x2,x3

# MLP to predict from outputs of 3 models
class EnsembleMLP(nn.Module):
    def __init__(self):
        super(EnsembleMLP, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x1,x2,x3):
        x = self.fc1(torch.tensor([x1,x2,x3]).cuda())
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x
