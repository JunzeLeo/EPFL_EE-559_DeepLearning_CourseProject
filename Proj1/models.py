
# In[]
# load packages

import torch
from torch import nn
from torch.nn import functional as F

# In[]

def percentageCorrect(prediction, target):
    
    return (prediction.argmax(dim = 1) == target).float().mean()


def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# In[]
# benchmark full connected network
"""
This class is an implemantation of a basic full connected network as benchmark.

"""
class basicFcnet(nn.Module):
 
    def __init__(self):
        nn.Module.__init__(self)
        
        self.fc1 = nn.Linear(392, 512)       
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)       
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)       
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 2)
        
        self.criterion  = nn.CrossEntropyLoss()
      
        
    def forward(self, x, **kwargs):
        
        x = x.flatten(start_dim=1)       
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.bn5(F.relu(self.fc5(x)))
        x = self.fc6(x)

        return x
    
    
    def loss(self, y, target):
        
        return self.criterion(y, target[0])


# In[]
# benchmark convnet

"""
This class is an implemantation of a basic convolutional network as benchmark.

The learning rate (step size) has to be tunned carefully as it affects the 
accuracy/error rate a lot. Pooling layers are not necessary as proved in some 
papers.

2e-3 is good for 81 nodes with batch size 100.
"""
class basicConvnet(nn.Module):
 
    def __init__(self):
        nn.Module.__init__(self)
        
        self.conv1 = nn.Conv2d(2, 8, 5)          # shape [8, 10, 10]
        self.bn2d1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)         # shape [16, 8,  8]
        self.bn2d2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 4)        # shape [32, 5,  5]
        self.bn2d3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 4)        # shape [32, 2,  2]
        self.bn2d4 = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32*2*2, 81)
        self.fc2   = nn.Linear(81, 2)
        
        self.criterion  = nn.CrossEntropyLoss()
      
        
    def forward(self, x, **kwargs):
        x = self.bn2d1(F.relu(self.conv1(x)))
        x = self.bn2d2(F.relu(self.conv2(x)))
        x = self.bn2d3(F.relu(self.conv3(x)))
        x = self.bn2d4(F.relu(self.conv4(x)))
        x = x.view(-1, 32*2*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    
    def loss(self, y, target):
        
        return self.criterion(y, target[0])
    
    
# In[]
# parallal convnet
"""
This class is an implemantation of a parallel convolutional network.

This network uses the two conv + fc networks to recongnize numbers seperately
and then feeding the concatenated two numbers to the final fc layer to compare
their values.

"""
class parallelConv(nn.Module):
 
    def __init__(self):
        nn.Module.__init__(self)
        
        self.conv1a = nn.Conv2d(1, 8, 5)          # shape [8, 10, 10]
        self.conv2a = nn.Conv2d(8, 16, 3)         # shape [16, 8,  8]
        self.conv3a = nn.Conv2d(16, 32, 4)        # shape [32, 5,  5]
        self.conv4a = nn.Conv2d(32, 32, 4)        # shape [32, 2,  2]
        self.fc1a   = nn.Linear(32*2*2, 64)
        self.fc2a   = nn.Linear(64, 10)
        
        self.conv1b = nn.Conv2d(1, 8, 5)          # shape [8, 10, 10]
        self.conv2b = nn.Conv2d(8, 16, 3)         # shape [16, 8,  8]
        self.conv3b = nn.Conv2d(16, 32, 4)        # shape [32, 5,  5]
        self.conv4b = nn.Conv2d(32, 32, 4)        # shape [32, 2,  2]
        self.fc1b   = nn.Linear(32*2*2, 64)
        self.fc2b   = nn.Linear(64, 10)
        
        self.fc3   = nn.Linear(20, 2)
        
        self.criterion  = nn.CrossEntropyLoss()
        
    def forward(self, x, is_vis : bool = False):
        
        x1 = x[:,0,:,:][:,None,:,:]
        y1 = F.relu(self.conv1a(x1))
        y1 = F.relu(self.conv2a(y1))
        y1 = F.relu(self.conv3a(y1))
        y1 = F.relu(self.conv4a(y1))
        y1 = y1.view(-1, 32*2*2)
        y1 = F.relu(self.fc1a(y1))
        y1 = self.fc2a(y1)
        
        x2 = x[:,1,:,:][:,None,:,:]
        y2 = F.relu(self.conv1b(x2))
        y2 = F.relu(self.conv2b(y2))
        y2 = F.relu(self.conv3b(y2))
        y2 = F.relu(self.conv4b(y2))
        y2 = y2.view(-1, 32*2*2)
        y2 = F.relu(self.fc1b(y2))
        y2 = self.fc2b(y2)
        
        y0 = self.fc3(torch.cat((F.relu(y1), F.relu(y2)), dim = 1))
        
        if is_vis:
            return y0
        else:
            return y0, y1, y2
    
    def loss(self, y, target, auxLoss: bool = True):
        if auxLoss:
            return self.criterion(y[0], target[0]) + self.criterion(y[1], target[1]) \
                   + self.criterion(y[2], target[2])
        else:
            return self.criterion(y[0], target[0]) 
    
# In[]
# siamese convnet
"""
This class is an implemantation of a basic siamese convolutional network.

This network uses the same conv + fc network (siamese) to recongnize numbers 
and then feeding the concatenated two numbers to the final fc layer to compare
their values. Three losses have the same weight, i.e. 1.

Learning rate 2e-3 is good for the current implmentation.
"""
class siamese(nn.Module):
 
    def __init__(self):
        nn.Module.__init__(self)
        
        self.conv1 = nn.Conv2d(1, 8, 5)          # shape [8, 10, 10]
        self.conv2 = nn.Conv2d(8, 16, 3)         # shape [16, 8,  8]
        self.conv3 = nn.Conv2d(16, 32, 4)        # shape [32, 5,  5]
        self.conv4 = nn.Conv2d(32, 32, 4)        # shape [32, 2,  2]
        self.fc1   = nn.Linear(32*2*2, 64)
        self.fc2   = nn.Linear(64, 10)
        self.fc3   = nn.Linear(20, 2)
        
        self.criterion  = nn.CrossEntropyLoss()
        
    def forward(self, x, is_vis : bool = False):
        
        x1 = x[:,0,:,:][:,None,:,:]
        y1 = F.relu(self.conv1(x1))
        y1 = F.relu(self.conv2(y1))
        y1 = F.relu(self.conv3(y1))
        y1 = F.relu(self.conv4(y1))
        y1 = y1.view(-1, 32*2*2)
        y1 = F.relu(self.fc1(y1))
        y1 = self.fc2(y1)
        
        x2 = x[:,1,:,:][:,None,:,:]
        y2 = F.relu(self.conv1(x2))
        y2 = F.relu(self.conv2(y2))
        y2 = F.relu(self.conv3(y2))
        y2 = F.relu(self.conv4(y2))
        y2 = y2.view(-1, 32*2*2)
        y2 = F.relu(self.fc1(y2))
        y2 = self.fc2(y2)
        
        y0 = self.fc3(torch.cat((F.relu(y1), F.relu(y2)), dim = 1))
        
        if is_vis:
            return y0
        else:
            return y0, y1, y2
    
    def loss(self, y, target, auxLoss: bool = True):
        if auxLoss:
            return self.criterion(y[0], target[0]) + self.criterion(y[1], target[1]) \
                   + self.criterion(y[2], target[2])
        else:
            return self.criterion(y[0], target[0]) 
    