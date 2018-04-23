import pandas as pd
import random
import math
import numpy as np
import copy

trainData = pd.read_csv("./data/train_ALL.csv")
trainData['emp_length'].fillna(trainData['emp_length'].median(), inplace=True)
trainData['annual_inc'].fillna(trainData['annual_inc'].median(), inplace=True)
trainData['title'].fillna(trainData['title'].median(), inplace=True)
trainData['pub_rec'].fillna(trainData['pub_rec'].median(), inplace=True)
trainData['revol_util'].fillna(trainData['revol_util'].median(), inplace=True)
trainData['total_acc'].fillna(trainData['total_acc'].median(), inplace=True)
trainData['collections_12_mths_ex_med'].fillna(trainData['collections_12_mths_ex_med'].median(), inplace=True)
# whole_list = np.array(trainData)

# labels = whole_list[18]  # 获得第一行（标签）
labels = trainData['acc_now_delinq'].tolist()
trainData = trainData.drop(['acc_now_delinq'])
trainData = np.array(trainData)


# print(labels)

def baggingData(dataSet):
    n, m = dataSet.shape

    # random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改
    features = random.sample(list(dataSet.columns.values[18]), m / 60)


# 最后一个属性还不能将样本完全分开，此时数量最多的label被选为最终类别
def majorClass(classList):
    classDict = {}
    for cls in classList:
        classDict[cls] = classDict.get(cls, 0) + 1
    sortClass = sorted(classDict.items(), key=lambda item: item[1])
    return sortClass[-1][0]


# 根据value(阈值)对连续变量划分数据集
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for i in dataSet:
        if i[featIndex] <= value:
            leftData.append(i)
        else:
            rightData.append(i)
    return leftData, rightData


# 去掉第i个属性，生成新的数据集
def splitData(dataSet, featIndex, features, value):
    newFeatures = copy.deepcopy(features)
    newFeatures.remove(features[featIndex])
    leftData, rightData = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= value:
            leftData.append(temp)
        else:
            rightData.append(temp)
    return newFeatures, leftData, rightData


# 计算Gini系数
def calcGini(dataSet, labels):
    labelCounts = {}
    for index in range(len(dataSet)):
        currentLabel = labels[index]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    Gini = 1
    for key in labelCounts:
        prob = labelCounts[key] / len(dataSet)
        Gini -= prob ** 2
    return Gini


def chooseBestFeature(dataSet, labels):
    bestGini = 1  # 基尼指数，1-segma(p^2),越小纯度越高
    bestFeatureIndex = -1
    bestSplitValue = None

    for i in range(len(dataSet[0])):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortFeatList = sorted(list(set(featList)))
        splitList = []
        for j in range(len(sortFeatList) - 1):
            splitList.append((sortFeatList[j] + sortFeatList[j + 1]) / 2)

        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            # gini = 1 - sigma(p^2)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            if newGini < bestGini:
                bestGini = newGini
                bestFeatureIndex = i
                bestSplitValue = splitValue
    return bestFeatureIndex, bestSplitValue


def createTree(input, features):
    classList = [item[18] for item in input]
    # 判断当前数据集中label的种类，若为一类则停止创建树，返回当前类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(features) == 1:
        # 若最后一个特征都没法分净，则选数量最多的作为当前类别
        return majorClass(classList)
    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet=input)
    bestFeature = features[bestFeatureIndex]
    # 生成新的去掉BestFeature的数据集
    newFeature, leftData, rightData = splitData(input, bestFeatureIndex, features, bestSplitValue)
