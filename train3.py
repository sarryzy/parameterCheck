import csv
import math
import os

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from typing import List
import math

# 使用L1Loss
def getTestList(idx:int):
    # 得到数据集中的第idx个数据
    fileName="data/csvName_{}.csv".format(idx)
    data=[]
    with open(fileName,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            if len(row)==2:
                data.append(float(row[1]))
    return data

def printGraph(idx:int,list):
    # 其中idx为对应的索引，data为对应的系数
    fileName = "data/csvName_{}.csv".format(idx)
    data = []
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 1:
                data.append(float(row[0]))
    x=torch.arange(0,150,1)
    y=data[0]/(1+(data[1]/x)**data[2])/(1+(x/data[3])**data[4])
    v=list[0]/(1+(list[1]/x)**list[2])/(1+(x/list[3])**list[4])
    plt.plot(x,y,'.')
    plt.plot(x,v,'.')
    plt.pause(0.001)
    plt.show()

class MyData(Dataset):
    def __init__(self):
        self.root_dir="./data"
        self.csvlist=os.listdir(self.root_dir)

    def __getitem__(self, idx):
        csvname=self.csvlist[idx]
        csvpath=os.path.join(self.root_dir,csvname)
        data=[];label=[]
        with open(csvpath,'r') as f:
            reader=csv.reader(f)
            for row in reader:
                if len(row)==1:
                    label.append(float(row[0]))
                else:
                    data.append(float(row[1]))
        data=torch.tensor(data,dtype=torch.float32)
        label=torch.tensor(label,dtype=torch.float32)
        label[0]=(label[0]-200)/300
        label[1]=(label[1]-5)/2
        label[2]=(label[2]-1.7)/0.2
        label[3]=(label[3]-140)/10
        label[4]=(label[4]-10)/20
        return data,label

    def __len__(self):
        return len(self.csvlist)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(150,256),
            nn.Linear(256,5)
        )

    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    device=torch.device("cpu")
    root_dir="data"
    idx=0
    dataset=MyData()
    dataloader=DataLoader(dataset,batch_size=1,num_workers=1)
    trainModel=MyModel()
    trainModel=trainModel.to(device)
    print(trainModel)
    loss_fn=nn.L1Loss()
    loss_fn=loss_fn.to(device)
    opt=torch.optim.Adam(trainModel.parameters())
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
        pre_loss=1000000
        now_loss=maxloss
        data=getTestList(idx)
        data=torch.tensor(data,dtype=torch.float32).to(device)
        data=trainModel.model(data)
        data[0] = 200 + 300 * data[0]
        data[1] = 5 + 2 * data[1]
        data[2] = 1.7 + 0.2 * data[2]
        data[3] = 140 + 10 * data[3]
        data[4] = 10 + 20 * data[4]
        print(now_loss)
        print(data)
        if now_loss<pre_loss:
            pre_loss=now_loss
            printGraph(idx, data.detach().numpy())
            torch.save(trainModel,"models/modelTrain3.pth")
