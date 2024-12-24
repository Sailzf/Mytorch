import torch 
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,3,5)
        self.pool1=nn.MaxPool2d(3,2)
        self.conv2=nn.Conv2d(3,3,3)
        self.pool1=nn.MaxPool2d(3,2)
        self.fc=nn.Linear(3*)
        self.conv2=