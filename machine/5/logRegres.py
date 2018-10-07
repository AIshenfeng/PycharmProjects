import matplotlib.pyplot as plt
from numpy import *
#从testSet.txt读取数据，创建data和label
def createdata():
    f=open('testSet.txt')
    data=[]
    label=[]
    for line in f.readlines():
        temp=line.strip().split('\t')
        temp_l=[1.0,float(temp[0]),float(temp[1])]
        data.append(temp_l)
        label.append(int(temp[2]))
    return data,label

#定义sigomid函数，用于分类
def sigomid(intX):
    z=1.0/(1.0+exp(-intX))
    return z

#训练获得最佳回归线参数，W=[w0,w1,~~~~wn]
def tran_W(data,label):
    #转换为Numpy矩阵
    data_m=mat(data)
    label_m=mat(label)
    #获得矩阵的形状
    m,n=shape(data_m)
    #初始化回归参数为（1,1,1，，，，，1）
    w=ones((n,1))
    #转置
    h=label_m.transpose()
    #最多的迭代次数
    max_C=500
    #步长
    alpha=0.001
    for i in range(max_C):
        error=(h-sigomid(data_m*w) )              #和实际分类的偏差
        w=w+alpha*data_m.transpose()*error      #修正
    return w

#print(tran_W(createdata()[0],createdata()[1]))

#画出这些点和分界线
def plotbestFit(weight):
    weights=mat(weight)
    data,label=createdata()
    X0=[];Y0=[]
    X1=[];Y1=[]

    n=shape(data)[0]
    for i in range(n):
        key=label[i]
        if key==1:
            X1.append(data[i][1])
            Y1.append(data[i][2])
        else:
            X0.append(data[i][1])
            Y0.append(data[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(X0,Y0,s=30,c='red',marker='s')
    ax.scatter(X1, Y1, s=30, c='green' )

    x=arange(-3.0,3.0,0.1)
    y=(-weight[0]-weight[1]*mat(x) )/weight[2]
    ax.plot(x,y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    plt.close()

data,labels=createdata()
plotbestFit(tran_W(data,labels))
