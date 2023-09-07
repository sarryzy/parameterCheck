import csv
import random
import torch
import shutil
import os
import numpy as np

shutil.rmtree("./data")
os.mkdir("./data")
# y=a0/(1+(a1/x)^a2)/(1+(x/a3)^a4)
# a0 (200,500),a1(5,7),a2(1.7,1.9),a3(140,150),a4(10,30)
log="./data"
for i in range(20000):
    a=np.random.rand(5)
    a[0]=200+300*a[0]
    a[1]=5+2*a[1]
    a[2]=1.7+0.2*a[2]
    a[3]=140+10*a[3]
    a[4]=10+20*a[4]
    a=list(a)
    fileName = "data/csvName_{}.csv".format(i)
    print(i)
    with open(fileName,'w',newline="") as fi:
        writer=csv.writer(fi)
        for j in range(len(a)):a[j]=round(a[j],2)
        print(str(a[0]))
        writer.writerow([str(a[0])])
        writer.writerow([str(a[1])])
        writer.writerow([str(a[2])])
        writer.writerow([str(a[3])])
        writer.writerow([str(a[4])])
        for x in range(0,150):
            y=a[0]/(1+(a[1]/x)**a[2])/(1+(x/a[3])**a[4])
            writer.writerow([x, round(y,2)])





