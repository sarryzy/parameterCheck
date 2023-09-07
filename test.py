import torch
from bihillParameterPrediction1.Utils.getData import *
from train4 import MyModel
idx=155
data=getTestList(idx)
data=torch.tensor(data,dtype=torch.float32)
trainModel=torch.load("models/modelTrain4.pth")
data=trainModel.model(data)
data=list(data.detach().numpy())
unnormalizeLabel(data)
printGraph(idx, data)

