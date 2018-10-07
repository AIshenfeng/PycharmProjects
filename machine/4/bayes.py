from numpy import *
import math

#创建实验样本，返回文章list,和文章对应的分类向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#根据文章数据集创建，词汇表并返回（set强制转换为list）
def createwordset(dataset):
    word_set=set([])
    for w in dataset:
        word_set=word_set | set(w)
    return list(word_set)

#文本转换为向量，向量长度为词汇表的长度，权重为1,0
def text2vec(word_list,input_text):
    vec=[0]*len(word_list)
    for k in input_text:
        if k in word_list:
            vec[word_list.index(k)]=1
    return vec

 #p(c|W)=p(Ci)*P(W|Ci)/P(W)

 #训练函数，得到文本集分类的P(C1) 和 P(W|Ci)的概率向量
def tran(tranmatrix,trancategory):
    #参数文本集对应的向量矩阵，和文本的类别向量
    n=len(tranmatrix)
    wordnums=len(tranmatrix[0])

    P1=sum(trancategory)/float(n)

    #初始化分子分母
    p0=ones(wordnums)
    p1=ones(wordnums)
    p0D=2.0
    p1D=2.0

    for k in tranmatrix:
        if trancategory[tranmatrix.index(k)]==0:
            p0+=k
            p0D+=sum(k)
        else:
            p1+=k
            p1D+=sum(k)
    return log(p0/p0D),log(p1/p1D),P1

 #分类函数
def classify(inputvec,p0,p1,P1):
    #参数分别为 待分类的文本向量，数据集的p0,p1概率向量，P1类别1的概率即P(C1)
    P_0=sum(inputvec*p0)+log(1-P1)
    P_1=sum(inputvec*p1)+log(P1)
    if P_0>P_1:
        return 0
    else:
        return 1

#分类测试
def test():
    dataset,classG=loadDataSet()
    word_list=createwordset(dataset)
    data_M=[]
    for tx in dataset:
        vec_temp=text2vec(word_list,tx)
        data_M.append(vec_temp)
    p0V, p1V, pAb=tran(data_M,classG)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(text2vec(word_list, testEntry))
    print (classify(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(text2vec(word_list, testEntry))
    print (classify(thisDoc, p0V, p1V, pAb))
import re
import random
#文件解析，以及垃圾邮件测试
def textsplit(str):
    L=re.split(r'\W*',str)
    return [k.lower() for k in L if len(k)>2]
def mail_test():
    doclist=[]
    wordlist=[]
    classlist=[]
    for k in range(1,26):
        a=textsplit(open('email/spam/%d.txt'%k,encoding='ISO-8859-15').read())
        doclist.append( (a))
        wordlist.extend(a)
        classlist.append(1)
        a=textsplit(open('email/ham/%d.txt'%k,encoding='ISO-8859-15').read())
        doclist.append(a)
        wordlist.extend(a)
        classlist.append(0)
    word_list=createwordset(doclist)
    test_set=[]
    test_class=[]
    rang=list(range(50))
    for n in range(10):
        num=int(random.uniform(0,len(rang)))
        test_set.append(text2vec(word_list,doclist[num]))
        test_class.append(classlist[num])
        del(rang[num])
    tran_M=[]
    tran_class=[]
    for id in rang:
        tran_M.append(text2vec(word_list,doclist[id]))
        tran_class.append(classlist[id])
    p_0,p_1,P1=tran(tran_M,tran_class)
    errorcnt=0
    for i in range(10):
        if classify(test_set[i],p_0,p_1,P1)!=test_class[i]:
            errorcnt+=1
    print('%f'%(float(errorcnt)/len(test_set)))


mail_test()