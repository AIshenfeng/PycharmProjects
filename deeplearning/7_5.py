# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


#定义参数
hidden=256
class_num=10
batch_size=100
learning_rate=0.01

Epoch=25
display=1



def craate_NN(x,weight,bias):
    layer1=tf.add(tf.matmul(x,weight["h1"]),bias["h1"])
    layer1_out=tf.nn.relu(layer1)

    layer2=tf.add(tf.matmul(layer1_out,weight["h2"]),bias["h2"])
    layer2_out=tf.nn.relu(layer2)

    pre=tf.add(tf.matmul(layer2_out,weight["out"]),bias["out"])


    return pre

X=tf.placeholder(tf.float32,[None,28*28])
Y=tf.placeholder(tf.float32,[None,class_num])

weights={
    "h1":tf.Variable(tf.truncated_normal([28*28,256])),
    "h2":tf.Variable(tf.truncated_normal([256,256])),
    "out":tf.Variable(tf.truncated_normal([256,10]))
}
bias={
    "h1":tf.Variable(tf.zeros([256])),
    "h2":tf.Variable(tf.zeros([256])),
    "out":tf.Variable(tf.zeros([10]))
}

output=craate_NN(X,weights,bias)

#loss
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output))
#err=tf.reduce_mean(tf.count_nonzero(output-Y))
#优化器
train=tf.train.AdamOptimizer(learning_rate).minimize(loss)


#启动Session()训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Epoch):
        ava_loss=0.0
        for i in range(np.int32(mnist.train.num_examples)/batch_size):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            batch_y=np.eye(class_num)[batch_y]
            _,Loss=sess.run([train,loss],feed_dict={X:batch_x,Y:batch_y})
            ava_loss+=Loss/(np.int32(mnist.train.num_examples)/batch_size)

        if  epoch%display==0:
            print ("epoch:",epoch,"loss=",ava_loss)
    print ("Finish!")

    accuracy=tf.equal(tf.argmax(output,axis=1),tf.argmax(Y,axis=1))
    accuracy=tf.reduce_mean(tf.cast(accuracy,tf.float32))
    print("Accuracy:",accuracy.eval({X:mnist.test.images,Y:np.eye(10)[mnist.test.labels]}))

