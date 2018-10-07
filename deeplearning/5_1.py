#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist =input_data.read_data_sets("MNIST_data/",one_hot=True)
print(mnist.train.images)
print(mnist.train.images.shape)

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.random_normal(([784,10])))
B=tf.Variable(tf.zeros([10]))

#前向传播

pre=tf.nn.softmax(tf.matmul(X,W)+B)

#反向传播
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pre),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#设置超参数
traning_epoch=25
batch_size=100
display_step=1

#这是Saver保存训练的模型
Saver=tf.train.Saver()
saver_path="log/model.ckpt"


#启动Session
with tf.Session() as sess:
    #初始化所有的变量
    sess.run(tf.global_variables_initializer())

    #循环训练25次
    for epoch in range(traning_epoch):
        avg=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
            avg+=c/total_batch
        if epoch%display_step==0:
            print("Epoch:",epoch+1,"lost=",avg)

    Saver.save(sess,saver_path)
    print("Finish!")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Saver.restore(sess,saver_path)

    #定义准确率的计算
    correct_prediction=tf.equal(tf.argmax(pre,1),tf.argmax(Y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("accuracy:",accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))

    output=tf.argmax(pre,1)
    batch_x,batch_y=mnist.train.next_batch(2)
    outputval,prdval=sess.run([output,pre],feed_dict={X:batch_x,Y:batch_y})
    print(outputval,prdval,batch_y)

