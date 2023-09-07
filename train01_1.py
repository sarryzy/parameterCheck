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
from Utils.getData import *

class MyData(Dataset):
    def __init__(self):
        self.root_dir="./data"
        self.csvlist=os.listdir(self.root_dir)

    def __getitem__(self, idx):
        csvname=self.csvlist[idx]
        csvpath=os.path.join(self.root_dir,csvname)
        data=[];label=[]
        with open(csvpath,'r') as f:
            k=1
            reader=csv.reader(f)
            for row in reader:
                if len(row)==1:
                    label.append(float(row[0]))
                elif len(row)==2 and  int(row[0])==k :
                    k+=10
                    data.append(float(row[1]))
        data=torch.tensor(data,dtype=torch.float32)
        label=normalizeLabel(label)
        label=torch.tensor(label,dtype=torch.float32)
        # label[0]=(label[0]-200)/300
        # label[1]=(label[1]-5)/2
        # label[2]=(label[2]-1.7)/0.2
        # label[3]=(label[3]-140)/10
        # label[4]=(label[4]-10)/20
        return data,label

    def __len__(self):
        return len(self.csvlist)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(15,16),
            nn.Linear(16,32),
            nn.Linear(32,5)
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
    # print(dataset[0])
    trainModel=trainModel.to(device)
    print(trainModel)
    loss_fn=nn.MSELoss()
    loss_fn=loss_fn.to(device)
    opt=torch.optim.Adam(trainModel.parameters())
    pre_loss=1000000
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
        data=getPoints(idx)
        print("data:",data)
        data=torch.tensor(data,dtype=torch.float32).to(device)
        data=trainModel.model(data)
        data=data.detach().numpy()
        print(data)
        data=unnormalizeLabel(data)
        print(now_loss)
        print(data)
        if now_loss<pre_loss:
            pre_loss=now_loss
            printGraph(idx,data)
            torch.save(trainModel,"models/modelTrainHowLong.pth")
