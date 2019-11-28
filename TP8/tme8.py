import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset


def store_approx_grad(var):
    def hook(grad):
        var.grad = grad
    return var.register_hook(hook)

class NNDropout(nn.Module) :
    def __init__(self):
        super(NNDropout,self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(784, 100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(100,10))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Softmax())
    
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            x = self.layers[i](x)
        return x


class CNNDropout():
    def __init__(self) :
        self.model = NNDropout()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.criterion = nn.CrossEntropyLoss() 


class NNBatchNorm(nn.Module) :
    def __init__(self):
        super(NNBatchNorm,self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(784, 100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,10))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Softmax())
    
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            x = self.layers[i](x)
        return x


class CNNBatchNorm():
    def __init__(self) :
        self.model = NNBatchNorm()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.criterion = nn.CrossEntropyLoss() 

class NNLayerNorm(nn.Module) :
    def __init__(self):
        super(NNLayerNorm,self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(784, 100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(100))
        self.layers.append(nn.Linear(100,10))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Softmax())
        self.cnt_forward = 0
        self.lay = [0,3,6]
        self.grad_layers = [0,3,6,9]
        self.grads = []
    
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            if i in self.grad_layers :
                x.requires_grad_()
                store_approx_grad(x)
                self.grads.append(x)
            x = self.layers[i](x)

        self.cnt_forward+=1
        return x


class CNNLayerNorm():
    def __init__(self) :
        self.model = NNLayerNorm()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.criterion = nn.CrossEntropyLoss() 



if __name__ == '__main__':
    dl = DataLoader(torchvision.datasets.MNIST('./', train=True, download=True))
    tensor = dl.dataset.data
    tensor  = tensor.to(dtype=torch.float32)
    tr = tensor.reshape(tensor.size(0), -1)
    tr = tr/128
    targets = dl.dataset.targets
    targets = targets.to(dtype=torch.long)

    x_train = tr[0:int(tr.size()[0]/20)]
    y_train = targets[0:int(tr.size()[0]/20)]

    x_valid = tr[int(tr.size()[0]/20):(int(tr.size()[0]/20)*2)]
    y_valid = targets[int(tr.size()[0]/20):(int(tr.size()[0]/20)*2)]
    
    x_test = tr[(int(tr.size()[0]/20)*2):]
    y_test = targets[(int(tr.size()[0]/20)*2):]

    bs=300

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    test_ds = TensorDataset(x_test,y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)


    loaders={}
    loaders['train'] = train_dl
    loaders['valid'] = valid_dl


    #Model
    cnn = CNNLayerNorm()
    for name, param in cnn.model.named_parameters() :
        print(f"name : {name} | param : {param.data.view(-1)}")

    #Tensorboard
    writer = SummaryWriter("runs/")
    cnt_loss_train = 0
    cnt_loss_eval = 0



    EPOCHS = 1001

    #Epoch
    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        print(f"Epoch {epoch}")

        cnn.model.train()
        for i, (data,target) in enumerate(loaders['train']):
            cnn.optimizer.zero_grad()
            output = cnn.model(data)
            loss = cnn.criterion(output,target)
            writer.add_scalars('Loss',{'train':loss}, cnt_loss_train)
            
            cnt_loss_train+=1
            loss.backward()
            cnn.optimizer.step()
        if epoch%50==0 :
            for name, param in cnn.model.named_parameters() :
                if name == "layers.0.weight":
                    writer.add_histogram('Weights/Layer1', param.data.view(-1), epoch//50)
                if name == "layers.3.weight":
                    writer.add_histogram('Weights/Layer2', param.data.view(-1), epoch//50)
                if name == "layers.6.weight":
                    writer.add_histogram('Weights/Layer3', param.data.view(-1), epoch//50)
            for i in range(len(cnn.model.grad_layers)) :
                writer.add_histogram(f"Grad/Layer{i+1}", cnn.model.grads[i].view(-1), epoch//50)
                    


        cnn.model.eval()
        for i, (data,target) in enumerate(loaders['valid']):
            output = cnn.model(data)
            loss = cnn.criterion(output,target)
            writer.add_scalars('Loss',{'eval':loss}, cnt_loss_eval)
            cnt_loss_eval+=1

        
    writer.close()
    

                    


        
        
