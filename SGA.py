#!/usr/bin/python
#-*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as ai
import numpy as np
import time

def loadData():                                 #文件读取函数 
    data=open('ex2data1.txt').readlines();                  
    l=len(data)                                 #mat为l*3的矩阵,元素都为0
    mat=zeros((l,3))                          
    index=0 
    xdata = ones((l,3))                         #xdata为l*3的矩阵，元素都为1              
    for line in data:
        line = line.strip().split(',')          #去除多余字符
        mat[index:,] = line[0:3]                #得到一行数据
        index +=1
    xdata[:,1] = mat[:,0]                       #得到第一列数据
    xdata[:,2] = mat[:,1]                       #得到第二列数据
    return xdata,mat[:,2]

def sigmoid(z):
    return 1.0/(1+exp(-z));

def gradAscent(xArr,yArr):
    xArr = mat(xArr)
    yArr = mat(yArr).transpose()
    maxCycles = 2000
    m,n = shape(xArr)
    weights = ones((n,1))
    weights_history1 = []
    weights_history2 = []
    weights_history3 = []
    iters = []
    i = 0
    for k in range(maxCycles):
        for j in range(m):
            alpha = 13/(1.0+k+j)+0.0002
            h = sigmoid(sum(xArr[j]*weights))
            error = yArr[j] - h
            error = float(error)
            weights += alpha * xArr[j].transpose()*error
            weights_history1.append(float(weights[0][0]))
            weights_history2.append(float(weights[1][0]))
            weights_history3.append(float(weights[2][0]))
            iters.append(i)
            i+=1
    
    return weights,weights_history1,weights_history2,weights_history3,iters


def show(xArr,yArr,weights):
    n = len(yArr)
    dataArr = array(xArr)
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    x = arange(25,100,1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    for i in range(n):
        if int(yArr[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='r',marker='o',label='Admitted')
    ax.scatter(xcord2,ycord2,s=30,c='g',marker='x',label='Not Admitted')
    ax.plot(x,y)
    plt.xlabel('Score1',size=25)                 # 横坐标
    plt.ylabel('Score2',size=25)                 # 纵坐标
    plt.title('Logistic',size=30)                  # 标题
    plt.show()  

def show1(weight1,weight2,weight3,iters):

    plt.figure(12)
    plt.subplot(221)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight1',size=10)
    plt.plot(iters,weight1,'ro')
    
   
    plt.subplot(222)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight2',size=10)
    plt.plot(iters,weight2,'bo')
    
    plt.subplot(212)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight3',size=10)
    plt.plot(iters,weight3,'go')

    plt.show()

xArr,yArr= loadData()
weight,weight1,weight2,weight3,iters= gradAscent(xArr,yArr)
show(xArr,yArr,weight)
show1(weight1,weight2,weight3,iters)
