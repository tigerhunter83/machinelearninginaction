from numpy import *
import os
import operator
from sklearn.datasets import fetch_openml
import time
import datetime as dt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights


# 绘制决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    print(m, n)
    alpha = 0.01
    weights = ones((n, 1))
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i].transpose()
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    import numpy as np
    m, n = shape(dataMatrix)
    weights = ones((n, 1))
    for j in range(numIter):
        print("iter %d" % j)
        dataIndex = list(range(m))
        #dataMatrix1 = dataMatrix[:, :]
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex].transpose()
            del(dataIndex[randIndex])
    return weights

# 改进的随机梯度上升函数
def stocGradAscent2(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 学习率逐渐降低
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

# 基于Sigmoid函数的分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 50)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

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
            Y1.append(0)
        elif Y[i] == '9':
            X_data1.append(X_data[i])
            Y1.append(1)

    print(len(X_data1))
    #split data to train and test
    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data1, Y1, test_size=0.9, random_state=42)
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    trainWeights = stocGradAscent1(array(X_train), y_train, 150)
    end_time = dt.datetime.now() 
    print('Stop learning {}'.format(str(end_time)))
    errorCount = 0
    targets = [1, 9]
    for i in range(len(X_test)):
        pred_Y = classifyVector(X_test[i],trainWeights)
        if pred_Y != y_test[i]:
            #print("not same.")
            errorCount += 1
        #else:
            #print("same")
        
    print("error rate is %f" % (float(errorCount)/ len(X_test)))




def main():

    # dataArr, labelMat = loadDataSet()
    # weights = stocGradAscent1(mat(dataArr), labelMat, 500)
    # print(weights)
    # plotBestFit(weights.getA())
    #multiTest()
    #colicTest()

    testmnist()

    

main()