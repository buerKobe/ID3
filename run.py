#encoding:utf-8
from trees import *
from treePlotter import *
import numpy as np
import pandas as pd
import sys

rank=[]
decision=True
def getData(filename):
    raw_train=pd.read_excel(filename,index_col=0)

    dataset=np.array(raw_train,dtype=str).tolist()

    label=np.array(raw_train.T.index,dtype=str)[:-1].tolist()

    obj=np.array(raw_train.T.index,dtype=str)[-1]
    
    return dataset,label,obj
dataset,label,obj=getData('train.xlsx')#读取表格数据
u=getUniqueVals(np.array(dataset),label)
myTree,rankTree=createTree(dataset,label,u,rank=rank)
createPlot(myTree) #得到决策树



