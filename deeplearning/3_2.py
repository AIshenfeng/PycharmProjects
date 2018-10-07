#coding=utf-8
import tensorflow as tf

#要获取那个变量,此处仍然需要定义一个变量
weight=tf.Variable(tf.random_normal([1]),name="weight")

#通过Saver() 载入模型
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"/home/shenfeng/123456")
    print("loss",sess.run(weight))
