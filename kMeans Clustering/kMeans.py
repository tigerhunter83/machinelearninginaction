from numpy import *
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:43:10 2020

@author: wangjingxian
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from numpy import array 
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mp1
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import csv
import os

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        #print(minJ)
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    
    return centroids

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    print(centroid0)
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1]= distMeas(mat(centroid0), dataSet[j,:])**2
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit:", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0]= bestCentToSplit
        print("the bestCentToSplit is ", bestCentToSplit)
        print("the len of bestClustAss is ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :]= bestClustAss
    return mat(centList), clusterAssment


def leakageAnalyze():
    dataset_path='data_in/'
    datasetlist=[]
    datasetlist=os.listdir(dataset_path)
    for i in range(len(datasetlist)):
        filename=datasetlist[i]
        rtu_id=filename.split('_')[0]
        loop=filename.split('_')[1]
        loop_id=loop.split('.')[0]
        
        path='data_in/'+filename#数据集读取，csv格式
        #print(data)
        path_train='data_train/'+filename
        data=pd.read_csv(path)
        data=data[-data.leakage_current.isin([0])]#删除漏电数据为0的所有行
        # X=data.leakage_current
        # X=X.values.reshape(-1,1)
        # data_number=X.shape[0]
        # print('除去漏电电流为0的数据，数据集使用的漏电数据条数为：',data_number)   
        # if data_number<1000:
        #     print('该条线路可用的有效数据量过少，低于1000条数据，参考性较小')

        #kmeans=KMeans(n_clusters=5).fit(data)#构建并训练模型
        myCentroids, clustAssing = kMeans(mat(data), 5)
    

# 球面距离计算及簇绘图函数
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(numClust=5):  # numClust：希望得到的簇数目
    datList = []
    # 获取地图数据
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        # 逐个获取第四列和第五列的经纬度信息
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # 绘图
    fig = plt.figure()
    # 创建矩形
    rect = [0.1, 0.1, 0.8, 0.8]
    # 创建不同标记图案
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 导入地图
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
def main():
    # dataMat = mat(loadDataSet('testSet.txt'))
    # myCentroids, clustAssing = kMeans(dataMat, 4)
    #leakageAnalyze()
    #dataMat3 = mat(loadDataSet('testSet2.txt'))
    #centList, myNewAssments = biKmeans(dataMat3, 3)
    #print(centList)
    clusterClubs(5)


main()

