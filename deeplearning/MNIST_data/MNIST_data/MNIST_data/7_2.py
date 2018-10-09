#coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def generate(sample_size, num_classes , diff,regression):
     #len(diff)
    np.random.seed(10)
    mean = np.random.rand(input_dim)
    cov = np.eye(input_dim)

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
lab_dim=3
lable_dim=3
Epoch=50
batch=25
train_size=1500
learning_rate=0.04



X,Y=generate(train_size,num_classes,[[3.0,3.0] ,[3.0,0]],False)
# colors=['r' if l[0]==1 else 'b' if l[1]==1 else 'y' for l in Y]
# plt.scatter(X[:,0],X[:,1],c=colors)
# plt.show()

#定义模型结构
input_feature=tf.placeholder(tf.float32,[None,input_dim])
input_lable=tf.placeholder(tf.float32,[None,lable_dim])

#定义参数变量
W=tf.Variable(tf.random_normal([input_dim,lable_dim]),name="weight")
b=tf.Variable(tf.zeros([lable_dim]), name="bais")

output=tf.matmul(input_feature,W)+b



#定义loss和err()准确率
cost=tf.nn.softmax_cross_entropy_with_logits(labels=input_lable,logits=output)
loss=tf.reduce_mean(cost,axis=0)

z=tf.argmax(tf.nn.softmax(output),axis=1)
sparse_label=tf.argmax(input_lable,axis=1)
err=tf.count_nonzero(z-sparse_label)        #错误的个数

#定义优化器
train=tf.train.AdamOptimizer(learning_rate).minimize(loss)

#启动Session训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(Epoch):
        arg_err=0.0
        for i in range(np.int32(len(Y)/batch)):
            x_temp=X[i*batch:(i+1)*batch,:]
            y_temp=Y[i*batch:(i+1)*batch,:]

            _,err_temp,out,Loss=sess.run([train,err,output,loss],feed_dict={input_feature:x_temp,input_lable:y_temp})
            arg_err+=err_temp*1.0/batch

        print("Epoch:",epoch,"  Loss=",Loss ,"   arg_err=",arg_err/(np.int32(len(Y)/batch)))
    print("Finish!")

#     train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
#     aa = [np.argmax(l) for l in train_Y]
#     colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
#     plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)
#
#     x = np.linspace(-1, 8, 200)
#
#     y = -x * (sess.run(W)[0][0] / sess.run(W)[1][0]) - sess.run(b)[0] / sess.run(W)[1][0]
#     plt.plot(x, y, label='first line', lw=3)
#
#     y = -x * (sess.run(W)[0][1] / sess.run(W)[1][1]) - sess.run(b)[1] / sess.run(W)[1][1]
#     plt.plot(x, y, label='second line', lw=2)
#
#     y = -x * (sess.run(W)[0][2] / sess.run(W)[1][2]) - sess.run(b)[2] / sess.run(W)[1][2]
#     plt.plot(x, y, label='third line', lw=1)
#
#     plt.legend()
#     plt.show()
    print(sess.run(W), sess.run(b))

#数据可视化


    train_X,train_Y=generate(300,3,[[3.0,3.0], [3.0,0]],False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)

    num_of_xs=200
    xs1=np.linspace(-2,8,num_of_xs)
    xs2=np.linspace(-2,8,num_of_xs)
    xx,yy=np.meshgrid(xs1,xs2)

    class_plant=np.zeros([num_of_xs,num_of_xs])
    for i in range(num_of_xs):
        for j in range(num_of_xs):
            class_plant[i,j]=sess.run(z,feed_dict={input_feature:[[xx[i,j],yy[i,j]]]})

    from matplotlib.colors import colorConverter,ListedColormap
    cmap=ListedColormap(
        [
            colorConverter.to_rgba('r',alpha=0.30),
            colorConverter.to_rgba('b', alpha=0.30),
            colorConverter.to_rgba('y', alpha=0.30)

        ]
    )

    plt.contourf(xx,yy,class_plant,cmap=cmap)
    plt.show()
    plt.close()


















