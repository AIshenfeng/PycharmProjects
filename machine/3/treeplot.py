import matplotlib.pyplot as plt
decisionNode =dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round",fc="0.8")
arrow_args=dict(arrowstyle="<-")
def plotNode(nodetext,centerPt,parentPt,nodeType):
    # 创建一个描述  annotate(s, xy, xytext=None, xycoords='data',textcoords='data', arrowprops=None, **kwargs)
    # s : 描述的内容
    # xy : 加描述的点
    # xytext : 标注的位置，xytext=(30,-30),表示从标注点x轴方向上增加30，y轴方向上减30的位置
    # xycoords 、textcoords :这两个参数试了好多次没弄明白，只知道 xycoords='data'给定就行，
    #  textcoords='offset points' 标注的内容从xy设置的点进行偏移xytext
    # textcoords='data' 标注内容为xytext的绝对坐标
    # fontsize : 字体大小，这个没什么好说的
    # arrowstyle : 箭头样式'->'指向标注点 '<-'指向标注内容 还有很多'-'
    # '->' 	head_length=0.4,head_width=0.2
    # '-[' 	widthB=1.0,lengthB=0.2,angleB=None
    # '|-|' 	widthA=1.0,widthB=1.0
    # '-|>' 	head_length=0.4,head_width=0.2
    # '<-' 	head_length=0.4,head_width=0.2
    # '<->' 	head_length=0.4,head_width=0.2
    # '<|-' 	head_length=0.4,head_width=0.2
    # '<|-|>' 	head_length=0.4,head_width=0.2
    # 'fancy' 	head_length=0.4,head_width=0.4,tail_width=0.4
    # 'simple' 	head_length=0.5,head_width=0.5,tail_width=0.2
    # 'wedge' 	tail_width=0.3,shrink_factor=0.5
    createPlot.ax1.annotate(s=nodetext,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
def createPlot(intree):

    fig=plt.figure(1,facecolor='white')
    fig.clf()               #创建新图形，清空绘图区
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plottree.W = getleafnums(intree)
    plottree.D = getHigh(intree)
    plottree.xoff=-0.5/plottree.W
    plottree.yoff=1.0
    plottree(intree,(0.5,1.0),'xxx')
    plt.show()

    plt.close()


#获取字典型树的叶子节点的个数
def getleafnums(tree):
    leafnums=0
    firstkey=list(tree.keys())[0]
    seconddict=tree[firstkey]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            leafnums+=getleafnums(seconddict[key])
        else:
            leafnums+=1
    return leafnums
#获得字典型树的高度
def getHigh(tree):
    maxhigh=0
    firstkey=list(tree.keys())[0]
    seconddict=tree[firstkey]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            thishigh=1+getHigh(seconddict[key])
        else:
            thishigh=1
        if thishigh>maxhigh:
            maxhigh=thishigh
    return maxhigh

#存储两棵树的数据
def retrievetree(i):
    listtree=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
              {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
    ]
    return listtree[i]



def plotMidtext(curpt,parpt,text):
    x=curpt[0]+float(curpt[0]+parpt[0])/2.0
    y=curpt[1]+float(curpt[1]+parpt[1])/2.0
    createPlot.ax1.text(x,y,text)

def plottree(intree,parpt,text):
    leafnums=getleafnums(intree)
    depth=getHigh(intree)
    firstkey=list(intree.keys())[0] #树的根节点
    curpt=(plottree.xoff+(1.0+float(leafnums))/2.0/plottree.W,plottree.yoff)
    plotMidtext(curpt,parpt,text)
    plotNode(firstkey,curpt,parpt,decisionNode)
    plottree.yoff=plottree.yoff-1.0/plottree.D
    secondtree=intree[firstkey]
    for key in secondtree.keys():
        if type(secondtree[key]).__name__=='dict':
            plottree(secondtree[key],curpt,str(key))
        else:
            plottree.xoff=plottree.xoff+1.0/plottree.W

            plotNode(secondtree[key],(plottree.xoff,plottree.yoff),curpt,leafNode)
            tempt=(plottree.xoff, plottree.yoff)
            plotMidtext(tempt, curpt, str(key))
    plottree.yoff=plottree.yoff+1.0/plottree.D


tree =retrievetree(0)
createPlot(tree)

