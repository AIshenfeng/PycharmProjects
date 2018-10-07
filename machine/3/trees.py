from math import log
#3-1 计算数据集的熵
#对数据集有一定的要求，数据集的最后一列必须是类别标签

def calcShannonEnt(dataset):

    m=len(dataset)                                  #实例的个数
    LabelsDict={}                                   #标签字典
    for temp in dataset:                               #统计每种标签各自有多少实例
        label=temp[-1]
        if label not in LabelsDict.keys():LabelsDict[label]=0
        LabelsDict[label]+=1
    ShannonEnt=0.0                                         #熵值

    for key in LabelsDict:
        p=float(LabelsDict[key])/m                        #p(label)  概率
        ShannonEnt-=p*log(p,2)

    return ShannonEnt                                   #返回熵



#3-2
#按照特定的特征划分数据集
#返回数据集中第axis个特征项的值为value的数据子集，并且数据子集中不包含这个特征项
def split_dataset(dataset,axis,value):
    #参数分别是：数据集，第几个特征，特征的值
    redataset=[]                            #因为dataset类型是引用，为了不改变原始数据，创建redataset
    for data_line in dataset:
        if data_line[axis]==value:
            redata_line=data_line[:axis]                #抽取第axis个特征值为value的数据，并且数据中剔除了第axis个特征
            redata_line.extend(data_line[axis+1:])
            redataset.append(redata_line)
    return  redataset                   #返回符合第axis个特征值为value的数据集

#3-3
# 选择最好的分类特征
#函数返回最好的特征的的下标
#先计算数据集的原始熵，然后按照每个特征项进行切分，计算切分后的熵。两者相减就是信息增益
#选择信息增益最大的特征项，并返回其下标
def choosebestfeature(dataset):
    baseShang=calcShannonEnt(dataset)           #记录数据集的原始熵
    n=len(dataset[0])-1                             #特征项的个数
    best_id=-1
    best_shang=0.0
    for i in range(n):                            #计算每个特征项的信息增益
        feature_values=[example[i] for example in dataset]      #第i个特征项的value
        unifeature_values=set(feature_values)                   #删除重复的值
        Shang=0.0
        for value in unifeature_values:                            #按照第i个特征项的值=value切分,得到子集的熵并按比例累加到Shang
            sub_dataset=split_dataset(dataset,i,value)
            prob=len(sub_dataset)/float(len(dataset))
            Shang+=(calcShannonEnt(sub_dataset)*prob)
        diffShang=baseShang-Shang                                       #按照第i个特征值切分的信息熵;因为切分后数据有规律不在杂乱，所以熵变小，原来的减去切分后的
        if diffShang>best_shang:
            best_shang=diffShang
            best_id=i
    return best_id                      #返回最好的特征的的下标

#3-4
#递归建立决策树
#递归结束条件：1.子集类别完全相同  2.所有的类别都已经使用过了
#针对第二种情况，如果子集的类别还是不相同，这时候采用多数表决方法，决定子集的类别

#多数表决算法，参数子集的的类别list
import operator
def majorclass(classlist):

    dict_temp={}
    for temp in classlist:
        dict_temp[temp]=dict_temp.get(temp,0)+1
    sorteddict=sorted(dict_temp.items(),key=operator.itemgetter(1),reverse=True)
    return sorteddict[0][0]

#创建树
def createTree(dataset,Labeles):
    #如果 子集类别相同，递归结束
    classlist=[example[-1] for example in dataset]
    if classlist.count(classlist[0])==len(dataset):
        return classlist[0]
    #如果所有的特征项都是用过了，递归结束
    if len(dataset[0])==1:
        return majorclass(classlist)
    #每次递归依据信息熵最高的特征项进行切分
    #获得最优的特征项和对应的标签
    bestfeatur=choosebestfeature(dataset)
    bestlabel=Labeles[bestfeatur]
    print(bestfeatur)
    print(bestlabel)
    print(dataset)
    print(Labeles)
    #创建树， 便签：下面的子树
    tree={bestlabel:{}}
    #获取此特征项对应的所有的值，放入set
    features=[example[bestfeatur] for example in dataset]
    unifeatures=set(features)
    del(Labeles[bestfeatur])


    #对每个值，获得相应的数据子集，然后递归建树
    for value in unifeatures:
        subLabels = Labeles[:]
        tree[bestlabel][value]=createTree(split_dataset(dataset,bestfeatur,value),subLabels)
    return tree

def createdata():
    dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

#分类测试
#参数是 创建好的字典树，数据集特征项对应的标签 ，待分类的list
def classify(inputree,Labels,test):
    first_key=list(inputree.keys())[0]
    secdict=inputree[first_key]
    first_index=Labels.index(first_key)                 #后去根节点特征项在Labels的index,即第几个特征项
    for key in secdict.keys():
        if test[first_index]==key:                      #按照 test的第index个特征项的值  往下分类。
            if type(secdict[key]).__name__=='dict':
                classLabel=classify(secdict[key],Labels,test)
            else:
                classLabel=secdict[key]
    return  classLabel

#通过序列化把树储存到磁盘文件上
def storetree(inputree,filename):
    import pickle
    fw=open(filename,'wb')

    pickle.dump(inputree,fw)
    fw.close()

#通过序列化从磁盘恢复树
def grabtree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

#字典树序列化存储测试
def tree_store_test():
    mydata,mylabels=createdata()
    mytree=createTree(mydata,mylabels)
    storetree(mytree,'tree_file.txt')
    re_tree=grabtree('tree_file.txt')
    print(classify(re_tree,mylabels,[1,1]))

#隐形眼镜的预测
def test():
    fr=open('lenses.txt','r')
    data=[line.strip().split('\t') for line in fr.readlines()]

    Labels=['age','prescript','astigmatic','tearRate']
    tree=createTree(data,Labels)
    print(tree)


test()

