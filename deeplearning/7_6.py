#coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def generate(sample_size, num_classes , diff,regression):
     #len(diff)
    input_dim=2
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


X,Y=generate(320,4,[[3.0,0],[3.0,3.0],[0,3.0]],True)
Y=Y%2
xa=[]
xb=[]
for l,k in zip(Y[:],X[:]):
    if l==0.0:
        xa.append(k)
    else:
        xb.append(k)

xa=np.array(xa)
xb=np.array(xb)

plt.scatter(xa[:,0],xa[:,1],c='r')
plt.scatter(xb[:,0],xb[:,1],c='b')
#plt.show()

#定义超参数
hidden_n=16
input_dim=2 #输入层节点的个数 input_n
output_n=1 #输出层节点个数
learn_ing=0.04
Epoch=3000
display=1000

#定义占位符
input_features=tf.placeholder(tf.float32,[None,input_dim])
input_lable=tf.placeholder(tf.float32,[None,output_n])

#定义参数
weights={
    "h1":tf.Variable(tf.truncated_normal([input_dim,hidden_n],stddev=0.1)),
    "h2":tf.Variable(tf.truncated_normal([hidden_n,output_n],stddev=0.1))
}

bias={
    "h1":tf.Variable(tf.zeros([hidden_n])),
    "h2":tf.Variable(tf.zeros([output_n]))

}

layer1=tf.nn.relu(tf.add( tf.matmul(input_features,weights["h1"]) , bias["h1"] ))
pred=tf.nn.tanh(tf.add( tf.matmul(layer1,weights["h2"]) ,bias["h2"]  ))



#定义loss
loss=tf.reduce_mean( (pred-input_lable)**2 + tf.nn.l2_loss(weights["h1"])*0.01+0.01*tf.nn.l2_loss(weights["h2"]))
#定义优化器
train=tf.train.AdamOptimizer(learn_ing).minimize(loss)
#定义err
err=tf.count_nonzero(input_lable-pred)



# '''
# 数据集
# '''
# X=[[0,0],[0,1],[1,0],[1,1]]
# Y=[[1],[0],[0],[1]]
# X=np.array(X).astype(np.float32)
# Y=np.array(Y).astype(np.int32)
#启动Session训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Epoch):
        _,pre,Loss,Error=sess.run([train,pred,loss,err],feed_dict={input_features:X,input_lable:np.reshape(Y,[-1,1])})
        if epoch%display==0:
            print("Epoch=",epoch,"  Loss=",Loss)
    #print(sess.run(pred,feed_dict={input_features:X}))


    #test
    X, Y = generate(320, 4, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
    Y = Y % 2
    _, pre, Loss, Error = sess.run([train, pred, loss, err],
                                   feed_dict={input_features: X, input_lable: np.reshape(Y, [-1, 1])})
    print("Loss=",Loss)
    xa = []
    xb = []
    for l, k in zip(Y[:], X[:]):
        if l == 0.0:
            xa.append(k)
        else:
            xb.append(k)

    xa = np.array(xa)
    xb = np.array(xb)

    plt.scatter(xa[:, 0], xa[:, 1], c='r')
    plt.scatter(xb[:, 0], xb[:, 1], c='b')



    from matplotlib.colors import colorConverter, ListedColormap
    nb_of_xs = 200
    xs1 = np.linspace(-3, 10, num=nb_of_xs)
    xs2 = np.linspace(-3, 10, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid
    # Initialize and fill the classification plane
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            # classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
            classification_plane[i, j] = sess.run(pred, feed_dict={input_features: [[xx[i, j], yy[i, j]]]})
            classification_plane[i, j] = int(classification_plane[i, j])

    # Create a color map to show the classification colors of each grid point
    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])
    # Plot the classification plane with decision boundary and input samples
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show()