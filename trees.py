#encoding:utf-8

from math import log
import operator
import numpy as np

leavesCnt=0
def entropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        imp=-prob*log(prob,2)
        ent += imp 
    return ent

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet,labels):
    numFeatures = len(dataSet[0]) - 1     
    #最后一行用于标签
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0; bestFeature = 0
    for i in range(numFeatures):       
        #遍历所有特征
        featList = [example[i] for example in dataSet]
        #为这个特征所有的样例创建一个列表
        uniqueVals = set(featList)      
        #得到一组独特的值
        newEntropy = 0.0
        print labels[i],'impurity'
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))

            sub_ent=entropy(subDataSet)
            print value,sub_ent
            newEntropy +=prob*sub_ent
        infoGain = baseEntropy - newEntropy   
        print 'information Gain',infoGain
        print 'Total impurity', newEntropy,'\n'
        if (infoGain > bestInfoGain):      
            #比较当前为止最大信息增益
            bestInfoGain = infoGain       
            #如果大于当前最大信息增益则设其为最大信息增益
            bestFeature = i
    return bestFeature,bestInfoGain

def findClass(classList,rank):
    global leavesCnt
    classCount={}
    for i in ['Yes','No']:
        classCount[i]=0
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    clas='Yes'
    for vote in classList:
        if classCount[vote]>classCount[clas]:
            clas=vote
    infos=[classCount['Yes'],classCount['No'],\
            (classCount['Yes']+1.0)/(classCount['Yes']+classCount['No']+2),\
            leavesCnt]
    leavesCnt+=1
    rank.append(infos)
    return clas,infos

def sortRankingTree(rankingTree,rank):
    sorted_rank=np.array(rank).T[-2:]
    idx=(-sorted_rank[0]).argsort()
    sorted_idx=sorted_rank[1][idx]
    sorted_prob=sorted_rank[0][idx]
    new_ranks=[1]
    for i in range(len(sorted_prob[1:])):
        if sorted_prob[i+1]==sorted_prob[i]:
            new_ranks.append(new_ranks[-1])
        else:
            new_ranks.append(new_ranks[-1]+1)
    ret_idx=sorted_idx.argsort().astype(int)
    ret_ranks=np.array(new_ranks)[ret_idx]
    return ret_ranks

def rankingTree2string(rankTree):
    firstStr = rankTree.keys()[0]
    secondDict = rankTree[firstStr]
    for valueOfFeat  in secondDict:
        item=secondDict[valueOfFeat]
        if isinstance(item, dict):
            rankingTree2string(item)
        else: 
            string=str(item[:2])+'\n'
            string+="{:2.1f}".format(item[-2]*100.0)+'%\n'
            string+='rank'+str(item[-1])
            secondDict[valueOfFeat]=string


def Rank(rankTree,rank):
    firstStr = rankTree.keys()[0]
    secondDict = rankTree[firstStr]
    for valueOfFeat  in secondDict:
        item=secondDict[valueOfFeat]
        if isinstance(item, dict):
            Rank(item,rank)
        else: 
            secondDict[valueOfFeat][-1]=rank[item[-1]]

def getUniqueVals(dataSet,labels):
    uniqueVals=dict()
    i=0
    for feature in dataSet.T[:-1]:
        uniqueVals[labels[i]]=set(feature)
        i+=1
    return uniqueVals

def createTree(dataSet,labels,ValsSet,node='root',rank=None):
    #检查纯度以及是否遍历所有特征
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList) or len(dataSet[0])==1:
        return findClass(classList,rank)
    print '================================================================'
    print node
    print '----------------------------------------------------------------'
    bestFeat,bestInfoGain = chooseBestFeatureToSplit(dataSet,labels)
    #if bestInfoGain==0:
    #    return findClass(classList,rank)
    bestFeatLabel = labels[bestFeat]
    print 'I choose',bestFeatLabel
    node+='->'+bestFeatLabel+'='
    myTree = {bestFeatLabel:{}}
    rankTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    #print bestFeat
    featValues = [example[bestFeat] for example in dataSet]
    #print featValues
    uniqueVals = set(featValues)
    if len(uniqueVals)!=len(ValsSet[bestFeatLabel]):
       # some value have no example
       for val in ValsSet[bestFeatLabel]:
           if val not in uniqueVals:
               l,votes=findClass(classList,rank)
               myTree[bestFeatLabel][val]=l
               rankTree[bestFeatLabel][val]=[0,0,.5,votes[-1]]
            
    for value in uniqueVals:
        #print value
        subLabels = labels[:]       
        myTree[bestFeatLabel][value],rankTree[bestFeatLabel][value]\
                = createTree(splitDataSet(dataSet, bestFeat, value),
                        subLabels,ValsSet,node=node+value,rank=rank)
    return myTree,rankTree 

def classify(inputTree,featLabels,testVec,ranking=False):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec,ranking)
    else: 
        if ranking==False:
            classLabel=valueOfFeat
        else:
            classLabel=valueOfFeat[-1]
    return classLabel
