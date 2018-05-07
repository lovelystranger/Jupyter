# -*-coding:utf-8 -*-

from numpy import *
import numpy as np
import pandas ad pd
from math import log
import operator


#计算数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	#给所有可能分类创建字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
		    labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #以2为底计算香农熵
    for key in labelCounts:
    	prob = float(labelCounts[key])/numEntries
    	shannonEnt -= prob*log(prob,2)
    return shannonEnt


#对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

#对连续变量划分数据集，direction规定划分的方向，
#决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet,axis,value,direction):
	retDataSet = []
	for featVec in dataSet:
		if direction == 0:
		    if featVec[axis] > value:
		    	reducedFeatVec = featVec[:axis]
		    	reduceFeatVec.extend(featVec[axis+1:])
		    	retDataSet.append(reducedFeatVec)
		    else:
		    	if featVec[axis] < value:
		    		reducedFeatVec = featVec[:axis]
		    		reducedFeatVec.extend(featVec[axis+1:])
		    		retDataSet.append(reducedFeatVec)
	return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet,lables):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	bestSplitDict = {}
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
        #对连续型特征进行处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int'
            #产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList)-1):
            	splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)

            bestSplitEntropy = 10000
            slen = len(splitList)
            #求用第j个候选划分点划分时，得到的信息熵，并记录 最佳划分点
            for j in range(slen):
            	value = splitList[j]
            	newEntropy = 0.0
            	subDataSet0 = splitContinuousDataSet(dataSet,i,value,0)
            	subDataSet1 = splitContinuousDataSet(dataSet,i,value,1)
            	prob0 = len(subDataSet0)/float(len(dataSet))
            	newEntropy += prob0*calcShannonEnt(subDataSet0)
            	prob1 = len(subDataSet1)/float(len(dataSet))
            	newEntropy += prob1*calcShannonEnt(subDataSet1)
            	if newEntropy < bestSplitEntropy:
            		bestSplitEntropy = newEntropy
            		bestSplit = j
            #用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        #对离散型特征进行处理
        else:
        	

