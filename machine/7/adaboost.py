# -*- coding: utf-8 -*-
__author__ = "shenfeng"
from numpy import *
from matplotlib import pyplot as plt
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#通过阈值比较进行分类，
def stumpClassify(datamatrix,dimen,threshVal,threshIneq):
    #参数：训练数据集矩阵，第几个特征，阈值，大于、小于符号
    retAraay=ones((shape(datamatrix)[0],1))
    if threshIneq=='lt':
        retAraay[datamatrix[:,dimen]<=threshVal]=-1
    else:
        retAraay[datamatrix[:,dimen]>threshVal]=-1
    return retAraay

#找到最佳单层决策树
def buildStump(dataarr,lebals,D):
    #参数：训练数据集矩阵，类别标签，样本权重
    dataMat=mat(dataarr)

    lebalMat=mat(lebals).transpose()
    numsteps=10.0
    beststump={}
    m,n=shape(dataMat)
    bestclassset=mat(zeros((m,1)))
    minerr=inf
    for i in range(n):  #对于每个向量循环
        Min=dataMat[:,i].min()
        Max=dataMat[:,i].max()
        stepsize=(Max-Min)/numsteps
        for j in range(-1,int(numsteps)+1):

            for Ineq in ['lt','gt']:
                threshVal=Min+float(j)*stepsize
                predictVals=stumpClassify(dataMat,i,threshVal,Ineq)
                error=mat(ones((m,1)))
                error[predictVals==lebalMat]=0
                errorWeight=D.T*error
                #print('dim : %d, thresh %.2f, threshIneq :%s , errweiget :%.3f' % (i, threshVal, Ineq, errorWeight))
                if errorWeight<minerr:
                    minerr=errorWeight
                    beststump['dien']=i
                    beststump['threshVal']=threshVal
                    beststump['threshIneq']=Ineq
                    bestclassset=predictVals.copy()

    return beststump,minerr,bestclassset

D=mat(ones((5,1))/5)
data,lebal=loadSimpData()
#print(buildStump(data,lebal,D))

import math
#训练过程，产生最终分类器
def adaboostTranDS(data,classLebals,numIt=40):
    #numIt是最终分类器所能包含的单层分类器的个数

    #存放单层分类器数组，最后返回
    Classarr=[]
    m=shape(data)[0]
    #样本权重
    D=mat(ones((m,1))/m)
    #累计数据点的类别估计值，起每个样本点的符号代表所属的类别
    aggclassEst=mat(zeros((m,1)))

    for i in range(numIt):
        #首先建立一个单层决策树
        beststump,err,bestclassSet=buildStump(data,classLebals,D)
        #print('D:',D.T)

        #计算当前决策树的系数alpha值，并加入到当前决策树beststump中
        alpha=float(0.5*log((1.0-err)/max(err,1e-16)))
        #把当前决策树加入到最终的分类算法的决策树数组中
        beststump['alpha']=alpha
        Classarr.append(beststump)
        #print('classSet:',bestclassSet.T)
        #更新样本权重D
        expon=multiply(-1*alpha*mat(classLebals).T,bestclassSet)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #累加 样本估计分类
        aggclassEst+=alpha*bestclassSet
        #print('aggclasserr: ',aggclassEst.T)
        aggerr=multiply(sign(aggclassEst)!=mat(classLebals).T,ones((m,1)))
        errrate=aggerr.sum()/m
        #print('total error :',errrate)
        if errrate==0.0:break
    return Classarr,aggclassEst


classifierarr,aggclassSet=adaboostTranDS(data,lebal)

#测试
def adatestify(datatoclass,classifierarr):
    m=shape(mat(datatoclass))[0]
    aggclass=mat(zeros((m,1)))
    for i in range(len(classifierarr)):
        Set=stumpClassify(mat(datatoclass),
                                classifierarr[i]['dien'],classifierarr[i]['threshVal'],
                                classifierarr[i]['threshIneq'])
        aggclass+=classifierarr[i]['alpha']*Set
        print(aggclass)
    return sign(aggclass)

print(adatestify([[0,0]],classifierarr))


#ROC曲线的绘制，以及AUC的计算
def plotRoc(prestrengths,classLabels):

    nums=sum(array(classLabels)==1.0)
    ystep=float(1)/float(nums)
    xstep=float(1)/float(len(classLabels)-nums)
    preindexs=prestrengths.argsort()
    cur=(1.0,1.0)
    ysum=0.0
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for  index in preindexs.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0.0;delY=ystep
        else:
            delX=xstep;delY=0.0
            ysum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('ROC  ')
    ax.axis([0,1,0,1])
    plt.show()
    plt.close()
    print('AUC:',ysum*xstep)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    f=open(fileName)
    numFeat = len(f.readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

data,lebal=loadDataSet('/home/shenfeng/PycharmProjects/untitled/horseColicTraining2.txt')
temp,aggclassSet2=adaboostTranDS(data,lebal,10)
print(data)
print(temp)
plotRoc(aggclassSet2.T,lebal)