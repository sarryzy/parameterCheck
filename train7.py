from train import MyData
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from typing import List
import math
from Utils.getData import *

# 更改标准值范围为（-1,1），并且使用深层神经网络,使用l1loss和新的优化器
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(150,128),
            nn.Linear(128,64),
            nn.Linear(64,32),
            nn.Linear(32,16),
            nn.Linear(16,8),
            nn.Linear(8,5),
        )

    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    device = torch.device("cpu")
    root_dir = "data"
    idx = 0
    dataset = MyData()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    trainModel = MyModel()
    trainModel = trainModel.to(device)
    print(trainModel)
    loss_fn = nn.L1Loss()
    loss_fn = loss_fn.to(device)
    opt = torch.optim.Adadelta(trainModel.parameters())
    pre_loss = 1000000
    for i in range(10000):
        print("正在迭代第{}次".format(i))
        trainModel.train()
        maxloss=0
        for data in dataloader:
            input,output=data
            input=input.to(device)
            output=output.to(device)
            trainOutput=trainModel.model(input)
            loss=loss_fn(output,trainOutput)
            maxloss=max(maxloss,loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        trainModel.eval()
        now_loss=maxloss
        data=getTestList(idx)
        data=torch.tensor(data,dtype=torch.float32).to(device)
        data=trainModel.model(data)
        data=list(data.detach().numpy())
        unnormalizeLabel(data)
        print("当前最小误差为",pre_loss)
        if now_loss < pre_loss:
            pre_loss = now_loss
            printGraph(idx, data)
            print("当前作图的误差为",now_loss)
            print(data)
            torch.save(trainModel, "models/modelTrain7.pth")