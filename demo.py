import torch
from train01_1 import MyModel
from Utils.getData import *
from scipy.stats import pearsonr
import numpy as np

trainModel=torch.load("models/modelTrain01_1.pth")
# a=[300, 5.32, 1.29, 146.30, 18.79]
# generateDemoCsv(a,0) # 生成数据
# filePath="file.csv"
# drawGraphOfDemo(filePath,trainModel)
while True:
    plt.close()
    print("======================")
    a=np.random.rand(5)
    a[0]=200+300*a[0]
    a[1]=5+2*a[1]
    a[2]=1.7+0.2*a[2]
    a[3]=140+10*a[3]
    a[4]=10+20*a[4]
    a=list(a)
    noise=np.random.rand(1)*5
    print("当前噪声：",noise.item())
    print("理论值：",a)
    generateDemoCsv(a,noise)
    filePath="file.csv"
    b=drawGraphOfDemo(filePath,trainModel)
    getListOfPointsAndDraw(a,'black')
    R=computeR2(a,b)
    R=round(R,2)
    r=grade(a,b)
    mark=(1-r)*R*100
    print("R²值为：",R)
    print("=====================")
    plt.title("R²:{},mark:{}".format(R,round(mark.item())),color='red')
    plt.show()
    plt.pause(5)



