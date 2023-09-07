import torch
from train01_1 import MyModel
from Utils.getData import *
from scipy.stats import pearsonr

cnt = 0
trainModel = torch.load("models/modelTrain01_1.pth")
print(trainModel)
for idx in range(20000):
    data=getPoints(idx)
    data=torch.tensor(data,dtype=torch.float32)
    data=trainModel.model(data)
    data=list(data.detach().numpy())
    unnormalizeLabel(data)
    a=getListOfPoints(data)
    b=getTestList(idx)
    r,p=pearsonr(a,b)
    print(idx)
    if r**2<0.95: cnt+=1
print("R²小于0.95的个数为：",cnt)