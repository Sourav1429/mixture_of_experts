# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:39:59 2025

@author: Sourav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class expert(nn.Module):
    def __init__(self,input_feat,hidden,output,h,w):
        super(expert,self).__init__()
        self.conv1 = nn.Conv2d(input_feat, hidden, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(hidden,hidden,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(hidden*h//4*w//4,hidden)
        self.fc = nn.Linear(hidden,output)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc(x))


BATCH_SIZE = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

network = expert(3, 128, 10, 32, 32)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam()

loss_list = []
N_epochs = 100
for epoch in range(N_epochs):
     combined_loss = 0
     for inputs,labels in train_loader:
         pred = network(inputs)
         loss = criterion(pred,labels)
         combined_loss = combined_loss + loss.item()
         optim.zero_grad()
         loss.backward()
         optim.step()
     loss_list.append(combined_loss)

print("Training complete")
path = "./single_expert_model"
torch.save(network, path)
    
         