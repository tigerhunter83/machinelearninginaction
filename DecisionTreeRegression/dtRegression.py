from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os
import operator
from sklearn.datasets import fetch_openml
import time
import datetime as dt

'''
    CART实现代码
'''


# 加载数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 每行数据映射成浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 切分数据集函数：数据集、切分特征、该特征的某个值
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 创建叶节点的函数：目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 总方差计算函数：目标变量的平方误差
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 将数据集格式化为X Y
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    # X Y用于简单线性回归，需要判断矩阵可逆
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 不需要切分时生成模型树叶节点
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    # 返回回归系数
    #print(ws)
    return ws


# 用来计算误差找到最佳切分
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    #print(ws)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 核心函数：找到最佳二元切分方式
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 容许误差下降值
    tolS = ops[0]
    # 切分的最少样本数
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            #print(mat0)
            #print(mat1)
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 树构建函数：数据集、建立叶节点函数、误差计算函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断输入是否为一棵树
def isTree(obj):
    # 判断为字典类型返回true
    return (type(obj).__name__ == 'dict')


# 返回树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 树的后剪枝待剪枝的树、剪枝所需的测试数据
def prune(tree, testData):
    # 确认数据集非空
    if shape(testData)[0] == 0: return getMean(tree)
    # 假设发生过拟合，采用测试数据对树进行剪枝
    # 左右子树非空
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 剪枝后判断是否还是有子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 判断是否merge
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果合并后误差变小
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def test4():
    myDat = loadDataSet('ex00.txt')

    myMat = mat(myDat)

    createTree(myMat)

    plt.plot(myMat[:, 0], myMat[:, 1], 'ro')

    plt.show()

    myDat1 = loadDataSet('ex0.txt')

    myMat1 = mat(myDat1)

    createTree(myMat1)

    plt.plot(myMat1[:, 1], myMat1[:, 2], 'ro')

    plt.show()


def test1():
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    print(len(testMat))
    myTree = createTree(trainMat, ops= (1, 20))
    print(myTree)
    yHat = createForeCast(myTree, testMat[:, 0])
    np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]

def test2():
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, modelLeaf, modelErr, ops= (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(myTree)
    np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]

def test3():
    myMat2 = np.mat(loadDataSet('exp2.txt'))
    print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))

def testmnist():
    images, targets = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X_data = images/255.0
    Y = targets

    X_data1 = []
    Y1 = []
    print(Y[:10])
    for i in range(len(X_data)):
        if Y[i] == '1':
            X_data1.append(X_data[i])
            Y1.append(-1)
        elif Y[i] == '9':
            X_data1.append(X_data[i])
            Y1.append(1)
    columns = shape(np.mat(X_data1))[1]
   

    #print(len(X_data2))
    #split data to train and test
    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data1, Y1, test_size=0.15, random_state=42)

    X_train2 = mat(ones((len(X_train), columns + 1)))
    X_train2[:, :columns] = X_train
    X_train2[:,columns] = mat(y_train).T

    X_test2 = mat(ones((len(X_test), columns + 1)))
    X_test2[:, :columns] = X_test
    X_test2[:,columns] = mat(y_test).T

    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    trainMat = np.mat(X_train2)
    testMat = np.mat(X_test2)
    myTree = createTree(trainMat, modelLeaf, modelErr, ops= (1, 100000))
    
    end_time = dt.datetime.now() 
    print('Stop learning {}'.format(str(end_time)))
    
    errorCount = 0
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(myTree)
    corr = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]
   
    print("the corr is: %f" % (corr))

def main():
    #test2()
    testmnist()


