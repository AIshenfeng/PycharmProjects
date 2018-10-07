# -*- coding: utf-8 -*-
__author__ = "shenfeng"

from numpy import *
import matplotlib.pyplot as plt
#数据导入函数
def loadData(file):
    f=open(file)
    data=[]
    lebals=[]
    for line in f.readlines():
        temp=line.strip().split('\t')
        n=len(temp)
        tempL=[]
        for i in range(n-1):
            tempL.append(float(temp[i]))
        data.append(tempL)
        lebals.append(float(temp[-1]))
    return data,lebals

#标准回归函数
def standregres(dataX,lebalY):
    X=mat(dataX)
    Y=mat(lebalY).T
    xtx=X.T*X
    if linalg.det(xtx)==0.0:
        print('矩阵不可逆')
        return
    w=xtx.I*(X.T*Y)
    return w

#测试函数
def test():
    data,lebal=loadData('ex0.txt')
    w=standregres(data,lebal)
    print('回归系数是：',w)

    xmat=mat(data)
    ymat=mat(lebal)
    yhat=xmat*w

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xmat[:,1].flatten().A[0],
               ymat.T[:,0].flatten().A[0])

    xcopy=xmat.copy()

    ax.plot(xcopy[:,1],yhat)

    plt.show()
    plt.close()

#test()

#局部加权线性回归函数
def lwlr(testarr,xarr,yarr,k=1.0):
    xMat=mat(xarr)
    yMat=mat(yarr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))

    #对当前的点testarr，计算权重中的高斯核

    for i in range(m):
        diff=testarr-xMat[i,:]
        weights[i,i]=exp(diff*diff.T/(-2.0*k**2))

    #计算回归系数
    xtx= xMat.T*(weights*xMat)
    if linalg.det(xtx)==0.0:
        print('矩阵不可逆')
        return

    ws=xtx.I*(xMat.T*(weights*yMat))

    return testarr*ws

def lwlrtest(testarr,xarr,yarr,k=1.0):
    m=shape(testarr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testarr[i],xarr,yarr,k)
    return yHat

xarr,yarr=loadData('ex0.txt')
yHat=lwlrtest(xarr,xarr,yarr,0.01)
xMat=mat(xarr)
#对xMat中的某一列进行排序，然后得到排序后的index序列
index=xMat[:,1].argsort(0)
#xMat[index]得到的是三维的矩阵；[:,0:]表示取所有的二维矩阵中的第0行，来组成一个二维矩阵
xsort=xMat[index][:,0,:]
#画图
fig=plt.figure()
ax=fig.add_subplot(111)
#绘制曲线
ax.plot(xsort[:,1],yHat[index])
#scaater绘制散点图
ax.scatter(xMat[:,1].flatten().A[0],mat(yarr).T.flatten().A[0],s=2,c='red')
plt.show()
plt.close()



































