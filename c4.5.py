# -*-coding:utf-8 -*-

from numpy import *
import numpy as np
import pandas as pd
from math import log
import operator
import matplotlib.pyplot as plt


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
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#对连续变量划分数据集，direction规定划分的方向，
#决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet,axis,value,direction):
	retDataSet = []
	for featVec in dataSet:
		if direction == 0:
		    if featVec[axis] > value:
		    	reducedFeatVec = featVec[:axis]
		    	reducedFeatVec.extend(featVec[axis+1:])
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
		if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
			sortfeatList = sorted(featList)
			splitList = []
			for j in range(len(sortfeatList)-1):
				splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)

			bestSplitEntropy = 10000
			slen = len(splitList)
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
				bestSplitDict[labels[i]] = splitList[bestSplit]
			infoGain = baseEntropy - bestSplitEntropy
		else:
			uniqueVals = set(featList)
			newEntropy = 0.0
			splitInfo = 0.0
            #计算该特征下每种划分的信息熵
			for value in uniqueVals:
				subDataSet = splitDataSet(dataSet,i,value)
				prob = len(subDataSet)/float(len(dataSet))
				newEntropy += prob*calcShannonEnt(subDataSet)
				splitInfo -= prob*log(prob,2)
				infoGain = (baseEntropy - newEntropy)/splitInfo
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
		bestSplitValue = bestSplitDict[labels[bestFeature]]
		labels[bestFeature] = labels[bestFeature] + '<=' +str(bestSplitValue)
		for i in range (shape(dataSet)[0]):
			if dataSet[i][bestFeature] <= bestSplitValue:
				dataSet[i][bestFeature] = 1
			else:
				dataSet[i][bestFeature] = 0
	return bestFeature	
#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
		return max(classCount)
        	
#主程序，递归产生决策树
def createTree(dataSet,labels,data_full,labels_full):
	classList = [example[-1] for example in dataSet] 
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet,labels)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	if type(dataSet[0][bestFeat]).__name__ == 'str':
		currentLabel = labels_full.index(labels[bestFeat])
		featValuesFull = [example[currentLabel] for example in data_full]
		uniqueValsFull = set(featValuesFull)
	del(labels[bestFeat])
	for value in uniqueVals:
		subLabels = labels[:]
		if type(dataSet[0][bestFeat]).__name__ == 'str':
			uniqueValsFull.remove(value)
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels,data_full,labels_full)
	if type(dataSet[0][bestFeat]).__name__ == 'str':
		for value in uniqueValsFull:
			myTree[bestFeatLabel][value] = majorityCnt(classList)
	return myTree



df = pd.read_csv('watermelon.csv')
data = df.values[:,1:].tolist()
data_full = data[:]
labels = df.columns.values[1:-1].tolist()
labels_full = labels[:]
myTree = createTree(data,labels,data_full,labels_full)
print(myTree)

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")


#计算树的叶子节点数量
def getNumLeafs(myTree):
	numLeafs = 0
	firstSides = list(myTree.keys())
	firstStr = firstSides[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

#计算树的最大深度
def getTreeDepth(myTree):
	maxDepth = 0
	firstSides = list(myTree.keys())
	firstStr = firstSides[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1+getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

#画节点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',\
		xytext = centerPt,textcoords = 'axes fraction',va = 'center',ha = 'center',\
		bbox = nodeType,arrowprops = arrow_args)

#画箭头上的文字
def plotMidText(cntrPt,parentPt,txtString):
	lens = len(txtString)
	xMid = (parentPt[0] + cntrPt[0])/2.0 
	yMid = (parentPt[1] + cntrPt[1])/2.0 
	createPlot.ax1.text(xMid,yMid,txtString,)

def plotTree(myTree,parentPt,nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstSides = list(myTree.keys())
	firstStr = firstSides[0]
	cntrPt = (plotTree.x0ff + (1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
	plotMidText(cntrPt,parentPt,nodeTxt)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict = myTree[firstStr]
	plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW
			plotNode(secondDict[key],(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
			plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,str(key))
	plotTree.y0ff = plotTree.y0ff + 1.0/plotTree.totalD

def createPlot(inTree):
	fig = plt.figure(1,facecolor = 'white')
	fig.clf()
	axprops = dict(xticks = [],yticks = [])
	createPlot.ax1 = plt.subplot(111,frameon = False,**axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.x0ff = -0.5/plotTree.totalW
	plotTree.y0ff = 1.0
	plotTree(inTree,(0.5,1.0),'')
	plt.show()

createPlot(myTree)
