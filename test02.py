import random

import matplotlib.pyplot as plt
import torch
from train import MyModel
from Utils.getData import *
from scipy.stats import pearsonr

trainModel=torch.load("models/modelTrain01_1.pth")
while True:
    idx=random.randint(0,20000-1)
    plt.close()
    data=getPoints(idx)
    data=torch.tensor(data,dtype=torch.float32)
    data=trainModel.model(data)
    data=list(data.detach().numpy())
    unnormalizeLabel(data)
    printGraph(idx,data)
    plt.pause(2)
    print(idx)
