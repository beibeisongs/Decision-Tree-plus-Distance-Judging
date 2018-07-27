# encoding=utf-8
# Date: 2018-7-23
# Description: A Complete Decision Tree codes


import numpy as np
import math
import operator
import csv


def createDataSet():
    """
    :param label: an One Dimension Array
    :param dataSet: A Multi-Dimension Matrix
    :return: dataSet, label
    """

    dataSet = []
    with open("iris.csv", "r") as csvfile:
        reader2 = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
        for item2 in reader2:
            for i in range(len(item2) - 1):
                item2[i] = float(item2[i])
            dataSet.append(item2)
    csvfile.close()
    

    label = ['特征一', '特征二', '特征三', "特征四"]


    """
    dataSet = [[1,1,'yes'],
                [1,1,'yes'],
                [1,0,'no'],
                [0,1,'no'],
                [0,1,'no'],]
    
    label = ['no surfacing','flippers']
    """

    return dataSet, label


def calcShannonEnt(dataSet):    # <Sample>: dataSet = <class 'list'>: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

    numEntries = len(dataSet)   # <Sample>: numEntries = 5
    labelCounts = {}
    for featVec in dataSet: # <Sample>: featVec = <class 'list'>: [1, 1, 'yes']
        currentLabel = featVec[-1]  # <Sample>: currentLabel = 'yes'
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # <Sample>: labelCounts = {'yes': 2, 'no': 3}

    shannonEnt = 0

    for key in labelCounts:
        shannonEnt = shannonEnt - (labelCounts[key]/numEntries)*math.log2(labelCounts[key]/numEntries)

    return shannonEnt


def splitDataSet(dataSet,axis,value):   # <Sample>: dataSet = <class 'list'>: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']] axis = 0 value = 0
    retDataset = []
    for featVec in dataSet:
        if featVec[axis] == value:
            newVec = featVec[:axis]
            newVec.extend(featVec[axis+1:]) # <Sample>: newVec = <class 'list'>: [1, 'no']
            retDataset.append(newVec)   # <Sample>: retDatset = <class 'list'>: [[1, 'no'], [1, 'no']]
    return retDataset

def chooseBestFeatureToSplit(dataSet):

    numFeatures = len(dataSet[0]) - 1   # <Sample>: numFeatures = 3 - 1 = 2
    bestInfoGain = 0
    bestFeature = -1
    baseEntropy = calcShannonEnt(dataSet)   # <Sample>: BaseEntropy = 0.9709505944546686

    for i in range(numFeatures):

        allValue = [example[i] for example in dataSet]  # <Sample>: allValue = <class 'list'>: [1, 1, 1, 0, 0]
        allValue = set(allValue)    # <Sample> allValue = set([1, 1, 1, 0, 0]) = [0, 1]
        newEntropy = 0
        for value in allValue:
            splitset = splitDataSet(dataSet,i,value)    # <Sample>: splitset = <class 'list'>: [[1, 'no'], [1, 'no']]
            newEntropy = newEntropy + len(splitset)/len(dataSet)*calcShannonEnt(splitset)
        infoGain = baseEntropy - newEntropy

    if infoGain > bestInfoGain:
        bestInfoGain = infoGain
        bestFeature = i

    return bestFeature

def majorityCnt(classList):

    classCount = {}

    for value in classList:
        if value not in classCount: classCount[value] = 0
        classCount[value] += 1

    classCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return classCount[0][0]


def createTree(dataSet,labels):
    """

    :param dataset: An Multi-Dimension Matirx
    :param label: The label of each sample
                    One Dimension Array
    :return: myTree
    """

    # You will find that classList contains the two options: "yes", "no"
    classList = [example[-1] for example in dataSet]    # <Sample>: <class 'list'>: ['yes', 'yes', 'no', 'no', 'no']

    labelsCopy = labels[:]  # <Sample>: <class 'list'>: ['no surfacing', 'flippers']

    if classList.count(classList[0]) == len(classList): # <Sample>: False
        return classList[0]

    if len(dataSet[0]) == 1:    # <Sample>: False
        return majorityCnt(classList)

    bestFeature = chooseBestFeatureToSplit(dataSet) # <Description>: variable "bestFeature" reflects to the idx of the features
    bestLabel = labelsCopy[bestFeature] # <Sample>: bestLabel = "flippers"

    global step_i
    step_i += 1

    myTree = {bestLabel:{}}
    print("Step",step_i, ":",myTree)

    featureValues = [example[bestFeature] for example in dataSet]   # <Sample>: featuresValues = {1, 1, 0, 1, 1}
    featureValues = set(featureValues)  # <Sample>: featureValues = {0, 1}

    del(labelsCopy[bestFeature])    # <Sample>: labelsCopy = <class 'list'>: ['no surfacing']

    for value in featureValues:
        subLabels = labelsCopy[:]
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
        print("Step", step_i, ":", myTree)

    return myTree


def find_shorttest_node(secondTree, testVec, featureId):

    process_i = 0

    for value in secondTree.keys():

        process_i += 1
        if process_i == 1:
            distance_temp = abs(testVec[featureId] - value)
            distance_min = distance_temp
            value_choice = value
        else:
            distance_temp = abs(testVec[featureId] - value)
            if distance_temp < distance_min:
                distance_min = distance_temp
                value_choice = value
    return value_choice


def classify(inputTree,featLabels,testVec):

    currentFeat = list(inputTree.keys())[0]
    secondTree = inputTree[currentFeat]

    try:
        featureId = featLabels.index(currentFeat)
    except ValueError as err: print('error ! ')

    try:
        value = find_shorttest_node(secondTree, testVec, featureId)
        if type(secondTree[value]).__name__ == 'dict':
            classLabel = classify(secondTree[value],featLabels,testVec)
        else:
            classLabel = secondTree[value]

        return classLabel

    except AttributeError:  print(secondTree)


if __name__ == "__main__":
    """
    :param dataset: An Multi-Dimension Matirx
    :param label: The label of each sample
                    One Dimension Array
    :param step_i(global): Record the tree branch division step id
    :param myTree: the Tree here is finished construction
    """

    global step_i
    step_i = 0

    dataset,label = createDataSet()

    myTree = createTree(dataset,label)

    """
    TestCode1:
        a = [1.7, 4.5, 5.8, 1.64]
        Used for iris dataset and the iris model created
    """
    """
    Test
    """
    a = [4.9,2.4,3.3,1]


    print(classify(myTree,label,a))