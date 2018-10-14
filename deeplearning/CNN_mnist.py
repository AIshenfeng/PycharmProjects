# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

#定义超参数
learning_rate=0.0001
epoch=10000
display=100
batch_size=100

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bais(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def covn2d(X,W):
    return tf.nn.conv2d(X,W,[1,1,1,1],padding='SAME')

def maxpool(X):
    return tf.nn.max_pool(X,[1,2,2,1],[1,2,2,1],padding='SAME')

#定义占位符
x=tf.placeholder(tf.float32,[None,28*28])
x_images=tf.reshape(x,[-1,28,28,1])
y=tf.placeholder(tf.float32,[None,10])

#第一层卷积
w1=weights([3,3,1,32])
b1=bais([32])
layer1=tf.add(covn2d(x_images,w1),b1)
l1=tf.nn.relu(layer1)
covn_maxpool_1=maxpool(l1)

#第二层卷积
w2=weights([5,5,32,64])
b2=bais([64])
covn_2=tf.nn.relu(covn2d(covn_maxpool_1,w2)+b2)
covn_maxpool_2=maxpool(covn_2)

#全连接层
full_x=tf.reshape(covn_maxpool_2,[-1,7*7*64])
w_full=weights([7*7*64,1024])
b_full=bais([1024])
out_full=tf.nn.relu(tf.matmul(full_x,w_full)+b_full)

#为全连接层添加dropout
keep_drop=tf.placeholder(tf.float32)
drop_full=tf.nn.dropout(out_full,keep_drop)

#softmax作为输出层
w_softmax=weights([1024,10])
b_softmax=bais([10])
output=tf.nn.softmax(tf.matmul(drop_full,w_softmax)+b_softmax)

#定义损失函数
cross_entropy=tf.reduce_mean(  -tf.reduce_sum(y*tf.log(output),reduction_indices=[1]))

#定义优化器
train=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#定义准确率
correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#训练
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(epoch):
    train_x,train_y=mnist.train.next_batch(batch_size)
    train.run(feed_dict={x:train_x,y:np.eye(10)[train_y],keep_drop:0.5})

    if i%display==0:
        train_accuracy=accuracy.eval(feed_dict={x:train_x,y:np.eye(10)[train_y],keep_drop:1.0})
        print ("epoch: %d    accuracy= %f"%(i,train_accuracy))

print "Finish!!!"

#测试
test_x=mnist.test.images
test_y=mnist.test.labels
test_accuracy=accuracy.eval(feed_dict={x:test_x,y:np.eye(10)[test_y],keep_drop:1.0})
print test_accuracy


























