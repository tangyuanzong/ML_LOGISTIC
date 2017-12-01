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

def cost(weight,x,y):
    weight = np.matrix(weight)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(x*weight)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*weight))) 
    return np.sum((first-second)/len(x))

def prdict(weight,x,y):
    l  =len(x) 
    c = 0
    for i in range(l):
          u = x[i][0]*weight[0][0]+x[i][1]*weight[1][0]+x[i][2]*weight[2][0]
          u = float(sigmoid(u)) 
          if (y[i]==1 and u >0.5 ) or (y[i]==0 and u <0.5 ):
             c+=1
          i +=1
    return float(c)/float(l)



def gradAscent(xArr,yArr):
    xArr = mat(xArr)
    yArr = mat(yArr).transpose()
    maxCycles = 1000
    m,n = shape(xArr)
    alpha = 0.004
    weights = zeros((n,1))
    weights_history1 = []
    weights_history2 = []
    weights_history3 = []
    cost_h = []
    iters = []
    i = 0
    print "初始代价为："
    print cost(weights,xArr,yArr)
    for k in range(maxCycles):
        for j in range(m):
            alpha = 1.3/(1.0+k+j)+0.0002
            h = sigmoid(sum(xArr[j]*weights))
            error = yArr[j] - h
            error = float(error)
            cost_h.append(float(cost(weights,xArr,yArr)))
            weights_history1.append(float(weights[0][0]))
            weights_history2.append(float(weights[1][0]))
            weights_history3.append(float(weights[2][0]))
            weights += alpha * xArr[j].transpose()*error
            iters.append(i)
            i+=1
    
    print "最终代价为: "
    print cost(weights,xArr,yArr)
    return weights,weights_history1,weights_history2,weights_history3,iters,cost_h


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

def show1(weight1,weight2,weight3,iters,cost_h):
    plt.figure(1)
    plt.subplot(221)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight1',size=10)
    plt.plot(iters,weight1,'bo')
    
   
    plt.subplot(222)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight2',size=10)
    plt.plot(iters,weight2,'bo')
    
    plt.subplot(223)
    plt.xlabel('isers',size=10)
    plt.ylabel('weight3',size=10)
    plt.plot(iters,weight3,'go')

    plt.subplot(224)
    plt.xlabel('isers',size=10)
    plt.ylabel('cost',size=10)
    plt.plot(iters,cost_h,'go')
    plt.show()

xArr,yArr= loadData()
weight,weight1,weight2,weight3,iters,cost_h= gradAscent(xArr,yArr)
print "求得的参数为："
print weight
print "某考生两次的考试成绩分别为45,85时，录取的概率为："
print prdict(weight,xArr,yArr)
show(xArr,yArr,weight)
show1(weight1,weight2,weight3,iters,cost_h)
