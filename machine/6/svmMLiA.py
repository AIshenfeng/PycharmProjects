# -*- coding: utf-8 -*-
import random
from numpy import *

#辅助函数
#加载文件，创建dataset和labelset
def loadDataset():
    dataset=[]
    labelset=[]
    f=open('testSet.txt')
    for line in f.readlines():
        temp=line.strip().split('\t')
        dataset.append([float(temp[0]),float(temp[1])])
        labelset.append(float(temp[2]))
    return dataset,labelset

#根据第一个变量随机选择第二个变量
def selectJrand(i,m):
    #i是第一个变量的下标，m是训练集两本的数目
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j

#Aj>H   Aj=H
#AJ<L   Aj=L
#else    Aj=Aj
def cliAlpha(aj,L,H):
    if aj>H:
        aj=H
    if aj<L:
        aj=L

    return aj


#简化版SMO算法实现
#1.参数为 dataset,labelset,常数C,容错率，最大迭代次数
#2.初始化矩阵为Numpy类型，b=0,a*=0,迭代计数=0
#3.在迭代次数内循环：
#   外循环 在range(m)内遍历一次选择第一变量
#       得到fxi;Ei
#       如果第i个变量误差较大，也就是有很大机会进行优化，则进行内循环：否则直接选择下一个 第一个变量
#       内循环  随机选择第二个变量j
#              得到fxj和Ej
#              这时得到 Ei,Ej    可以计算得到ai(new)  aj(new)
#              也可以得到bi,bj的值     并跟新b值
def samplesmo(dataset,labelset,C,toler,maxiter):
    data=mat(dataset)
    label=mat(labelset).transpose()
    b=0
    m,n=shape(data)
    alpha=zeros((m,1))
    iter=0
    while iter<maxiter:
        #alphachange用来标记是否有优化
        alphachange=0
        for i in range(m):
            fxi=float(multiply(alpha,label).T*(data*data[i,:].T))+b
            Ei=fxi-float(label[i])
            if(label[i]*Ei<-toler and alpha[i]<C) or (label[i]*Ei>toler and alpha[i]>C):
                j=selectJrand(i,m)
                fxj=float(multiply(alpha,label).T*(data*data[j,:].T))+b
                Ej=fxj-float(label[j])

                #得到了Ei,Ej;计算aj(new)  如果aj有足够的变化，计算ai(new)
                alphaIold=alpha[i].copy()
                alphaJold=alpha[j].copy()
                #因为ai*yi+aj*yj=一定值，所以ai和aj 是线性关系。根据yi和yj的关系取得aj的取值范围L~H
                if label[i]!=label[j]:
                    L=max(0,alpha[j]-alpha[i])
                    H=min(C,C+alpha[j]-alpha[i])
                else:
                    L=max(0,alpha[j]+alpha[i])
                    H=min(C,alpha[j]+alpha[i]-C)
                #如果L==H,则选取的i,j是不能得到优化，放弃选择下一个i
                if L==H:
                    print("L==H")
                    continue
                #eta=2Kij-Kii-Kjj
                eta=2.0*data[i,:]*data[j,:].transpose()-data[i,:]*data[i,:].transpose()-data[j,:]*data[j,:].transpose()
                #如果eta>=0,不能得到优化
                if eta>=0:
                    print("eta>=0")
                    continue
                #得到不考虑约束条件的aj
                alpha[j]=alpha[j]-label[j]*(Ei-Ej)/eta
                #在约束条件下得到新的aj
                alpha[j]=cliAlpha(alpha[j],L,H)
                #如果aj有足够的优化，就计算ai(new),否则舍弃，选择下个i变量
                if abs(alpha[j]-alphaJold)<0.00001:
                    print("j is not enough ")
                    continue
                alpha[i]=alpha[i]+label[j]*label[i]*(alphaJold-alphaIold)
                #最后要更新b,b的钢芯要借助i,j对应的bi,bj.
                b1=b-Ei-label[i]*(alpha[i]-alphaIold)*data[i,:]*data[i,:].T-label[j]*(alpha[j]-alphaJold)*data[j,:]*data[j,:]
                b2=b-Ej-label[i]*(alpha[i]-alphaIold)*data[i,:]*data[j,:].T-label[j]*(alpha[j]-alphaJold)*data[j,:]*data[j,:]
                if (0<alpha[i] )and (C>alpha[i]):b=b1
                elif (0<alpha[j] )and (C>alpha[j]):b=b2
                else: b=(b1+b2)/2.0
                alphachange+=1
                print('iter : %d i: %d,pairs changed %d'%(iter,i,alphachange))
        #如果没有优化，迭代次数加一。否则iter=0；直到在所有的变量上都没有优化并且迭代500次，循环结束。
        if alphachange==0:
            iter+=1
        else:
            iter=0
        print('itertion number %d'%iter)
    return b,alpha

#完整的Platt SMO算法加速优化
#完整版中alpha的更改和代数计算一样，不一样的只是选择alpha的方式不同
#在完整版中采用了启发式方法。
#选择过程在两种方式之间交替进行，1.对所有的数据进行单遍扫描 2.对alpha不等于0或者C的非边界处扫描（要建立这些alpha值的列表）
#选择第二个变量时，要通过最大步长的方式进行。所以要建立一个误差缓存表，用于存放Ei。


#辅助函数
#定义一个数据结构用来保存所有的关键值
class optStruct:
    def __init__(self,dataMatIn,labelMat,C,toler):
        self.X=dataMatIn
        self.labelclass=labelMat
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.Cache=mat(zeros((self.m,2))) #第一列给出的是是否有效的标志位,1表示有效，第二列给出具体的数值Ei

#根据给的alpha[k],给出Ek值
def calcEk(Os,k):
    fxk=float(multiply(Os.alphas,Os.labelclass).T*(Os.X*Os.X[k,:].T)+Os.b)

    return fxk-float(Os.labelclass[k])

#根据第一个变量ai，和Ei选择第二个变量aj,    并返回j,和Ej
#第二个变量的选择标准是最大步长，|Ek-Ei|最大的k 作为第二个变量
def selectJ(i,Os,Ei):

    maxk=-1
    maxdelete=0
    Ej=0
    Os.Cache[i]=[1,Ei]
    validCachelist=nonzero(Os.Cache[:,0].A)[0]

    if (len(validCachelist))>1:

        for k in validCachelist:
            if k==i:
                continue
            Ek=calcEk(Os,k)
            deleteEk=abs(Ek-Ei)
            if deleteEk>maxdelete:
                maxk=k
                maxdelete=deleteEk
                Ej=Ek
        return maxk,Ej
    else:
        j=selectJrand(i,Os.m)

        Ej=calcEk(Os,j)
    return j,Ej

#更新Ek
def updateEk(Os,k):
    Ek=calcEk(Os,k)
    Os.Cache[k]=[1,Ek]

#完整的Platt SMO优化算法之内循环
def innerL(i,os):

    Ei=calcEk(os,i)

    #如果i的误差很大，说明有优化的可能，就往下执行
    if ((os.labelclass[i]*Ei<-os.tol)and(os.alphas[i]<os.C))\
            or((os.labelclass[i]*Ei>os.tol)and(os.alphas[i]>0)):
        j,Ej=selectJ(i,os,Ei)

        alphaIold=os.alphas[i].copy()
        alphaJold=os.alphas[j].copy()
        if(os.labelclass[i]!=os.labelclass[j]):
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i]-os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta=2.0*os.X[i,:]*os.X[j,:].T\
            -os.X[i,:]*os.X[i,:].T\
            -os.X[j,:]*os.X[j,:].T
        if eta>=0:
            print("eta>=0")
            return 0
        os.alphas[j]-=os.labelclass[j]*(Ei-Ej)/eta
        os.alphas[j]=cliAlpha(os.alphas[j],L,H)

        updateEk(os,j)
        if (abs(os.alphas[j]-alphaJold)<0.00001):
            print("j not enough")
            return 0

        os.alphas[i]+=os.labelclass[j]*os.labelclass[i]*(alphaJold-os.alphas[j])
        updateEk(os,i)
        b1=os.b-Ei\
           -os.labelclass[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[i,:].T\
           -os.labelclass[j]*(os.alphas[j]-alphaJold)*os.X[i,:]*os.X[j,:].T
        b2=os.b-Ei\
           -os.labelclass[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[j,:].T\
           -os.labelclass[j]*(os.alphas[j]-alphaJold)*os.X[j,:]*os.X[j,:].T

        if (0<os.alphas[i])and (os.C>os.alphas[i]):os.b=b1
        elif (0<os.alphas[j])and (os.C>os.alphas[j]):os.b=b2
        else:os.b=(b1+b2)/2.0
        return 1
    else:

        return 0

#完整版platt_SMO算法的外循环
#参数和简化版本一样
def smo_p(datamat,labelmat,c,toler,maxint,ktup=('lin',0)):
    os=optStruct(mat(datamat),mat(labelmat).transpose(),c,toler)

    #对循环（迭代）计数
    iter=0
    alphaChanged=0
    #用来在下面两种方式中切换
    #  True: 1.所有数据中遍历   False: 2.在alpha非边界值中遍历
    entirset=True
    while (iter<maxint) and ((alphaChanged>0) or (entirset)):

        #alphaChanged记录每次迭代（循环）pairs changed 的个数
        alphaChanged=0
        #在所有数据上遍历，作为外循环的第一个变量
        if entirset:

            for i in range(os.m):
                alphaChanged+=innerL(i,os)
                print('fullset ,itre %d i:%d changed %d'%(iter,i,alphaChanged))
            iter+=1
        #遍历非边界alpha值
        else:
            #建立所有非边界值对应下标的list
            noboundlist=nonzero((os.alphas.A>0) *  (os.alphas.A<c))[0]

            for i in noboundlist:
                alphaChanged+=innerL(i ,os)
                print('nobound ,iter %d i:%d changed %d'%(iter,i,alphaChanged))
            iter+=1

        #是迭代（循环）在下面两种方式中遍历
            # 1.所有数据中遍历  2.在alpha非边界值中遍历
        if entirset:
            entirset=False_#如果上一次在所有中遍历，则转到第二种方式中
        elif alphaChanged==0:
            entirset=True   #如果上次在非alpha边界值处遍历，但是没有一次优化改变，则转换到第一种方式：在所有数据上遍历

        print('iter %d'%iter)

    return os.b,os.alphas
data,label =loadDataset()

b,alphas=smo_p(data,label,0.6,0.001,40)
print(b)

print(alphas[alphas>0])

def calcw(alphas,data,classl):
    x=mat(data)
    lebal=mat(classl).transpose()
    m,n=shape(x)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*lebal[i],x[i,:].T)
    return w



w=calcw(alphas,data,label)
dataMat=mat(data)
error_cnt=0
print(w)
for i in  range(len(dataMat)):
    if int(sign(dataMat[i]*mat(w)+b))!=int(label[i]):
        error_cnt+=1
print(error_cnt)



