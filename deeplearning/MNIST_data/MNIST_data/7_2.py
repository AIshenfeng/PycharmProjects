#coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def generate(sample_size, mean, cov, diff,regression,num_classes = 3):
     #len(diff)
    samples_per_class = int(sample_size/num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    if regression==False: #one-hot  0 into the vector "1 0
        Y=Y0.astype(int)
        Y=np.eye(num_classes)[Y]
        Y0=Y
    X, Y = shuffle(X0, Y0)

    return X,Y

input_dim=2
num_classes=3
lable_dim=3
Epoch=50
batch=25
train_size=1500

np.random.seed(10)
mean=np.random.rand(input_dim)
cov=np.eye(input_dim)
X,Y=generate(train_size,mean,cov,[[3.0,3.0] ,[3.0,0]],False)
# colors=['r' if l[0]==1 else 'b' if l[1]==1 else 'y' for l in Y]
# plt.scatter(X[:,0],X[:,1],c=colors)
# plt.show()

#定义模型结构

