
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
from torchvision.models import resnet18



class custom_resnet18(nn.Module):

    def __init__(self, num_classes=257, p=0.5, MCD = False):
        super(custom_resnet18, self).__init__()
        self.p = p
        self.MCD = MCD
        self.num_classes = num_classes
        #Original Resnet Architecture
        self.resnet=resnet18(pretrained=True)
        """
        for child in (list(self.resnet.children())[0:-3]):
            #print(idx, child)
            for param in child.parameters():
                param.requires_grad = False
        """

        self.resnet = nn.Sequential(
                    *list(self.resnet.children())[0:-1]
                )

        self.D0 = nn.Dropout(p)
        self.fc1 = nn.Linear(512,256)
        self.D1 = nn.Dropout(p)
        self.fc2 = nn.Linear(256,256)
        self.D2 = nn.Dropout(p)
        self.fc3 = nn.Linear(256,num_classes)

        self.dropout_layers = [self.D0, self.D1, self.D2]
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = self.resnet(x)
        x = x.view(-1,512)
        if self.MCD: x = self.D0(x)
        x = self.fc1(x)
        if self.MCD: x = self.D1(x)
        x = self.fc2(x)
        if self.MCD: x = self.D2(x)
        x = self.fc3(x)

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


