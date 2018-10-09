#coding=utf-8
import tensorflow as tf
import numpy as np

#定义超参数
hidden_n=16
input_dim=2 #输入层节点的个数 input_n
output_n=1 #输出层节点个数
learn_ing=0.04
Epoch=140
display=20

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
loss=tf.reduce_mean( (pred-input_lable)**2 )
#定义优化器
train=tf.train.AdamOptimizer(learn_ing).minimize(loss)
#定义err
err=tf.count_nonzero(input_lable-pred)



'''
数据集
'''
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[1],[0],[0],[1]]
X=np.array(X).astype(np.float32)
Y=np.array(Y).astype(np.int32)
#启动Session训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Epoch):
        _,pre,Loss,Error=sess.run([train,pred,loss,err],feed_dict={input_features:X,input_lable:Y})
        if epoch%display==0:
            print("Epoch=",epoch,"  Loss=",Loss)
    print(sess.run(pred,feed_dict={input_features:X}))