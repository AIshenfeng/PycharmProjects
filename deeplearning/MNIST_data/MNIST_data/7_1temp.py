#coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle




#模拟数据点
def generate(sample_size, mean, cov, diff,regression):
    num_classes = 2 #len(diff)
    samples_per_class = int(sample_size/2)

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


input_dim = 2
np.random.seed(10)
num_classes =2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [[3.0,3.0]],False)
#print(X,"\n",Y)
colors = ['r' if l[0] == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()
lab_dim = num_classes

#定义占位符
input_features=tf.placeholder(tf.float32,[None,input_dim])
input_labels=tf.placeholder(tf.int32,[None,num_classes])

#定义学习参数
W=tf.Variable(tf.random_normal([input_dim,lab_dim]),name="Weight")
b=tf.Variable(tf.zeros([lab_dim]),name="Bais")

#定义output
output=tf.matmul(input_features,W)+b
z=tf.nn.softmax(output)
pred=tf.argmax(z,axis=1)

#定义损失函数(sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)logits 的长度需要比 labels 多一维，就可以了。)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels,logits=output))

#定义错误率：

lable=tf.argmax(input_labels,axis=1)
err=tf.count_nonzero(pred-lable)

#定义学习率
learning_rate=0.04

#定义优化器
train=tf.train.AdamOptimizer(learning_rate).minimize(cost)



#定义Epoch和display和batchsize
Epoch=50
display=2
batchsize=25



#启动Sesson训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(Epoch):
        avg_err=0.
        lost=0.
        for j in range(np.int32(len(Y)/batchsize)):
            x_temp=X[j*batchsize:(j+1)*batchsize,:]
            y_temp=Y[j*batchsize:(j+1)*batchsize,:]

            _,lost_temp,err_temp,pre,y=sess.run([train,cost,err,pred,lable],feed_dict={input_features:x_temp,input_labels:y_temp})
            # print(pre)
            # print(y)
            # print(err_temp)
            # print(err_temp*1.0/batchsize)
            avg_err+=err_temp*1.0/batchsize
            lost+=lost_temp/(np.int32(len(Y)/batchsize))

        if i%display==0:
            print("Epoch:",i,"Lost=",lost,"avg_err=",avg_err/(np.int32(len(Y)/batchsize)))
    print("Finish!!")













