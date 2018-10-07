#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moving_avg(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx <w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]



train_X=np.linspace(-1,1,100)
train_Y=2.0*train_X+np.random.rand(train_X.size)*0.3



#创建模型
#占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")
#模型参数
weight=tf.Variable(tf.random_normal([1]),name="weight")
bais=tf.Variable(tf.zeros([1]),name="bais")
#前向结构
z=tf.multiply(X,weight)+bais
#将预测值以直方图的方式显示
tf.summary.histogram('z',z)
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))     #代价函数 生成值和真实值之间的平方差
tf.summary.scalar('loss_function',cost) #损失以标量方式显示
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #GradientDescentOptimizer是封装好的梯度下降算法。


#模型建立好后 tensorflow 通过session来进行
init=tf.global_variables_initializer()#初始化所有的变量
#定义参数
training_epochs=20
display_step=2

#通过saver保存模型
saver=tf.train.Saver()

#启动session
with tf.Session() as sess:
    sess.run(init)          #初始化所有变量
    merged_op=tf.summary.merge_all()#合并所有的summery
    summery_writer=tf.summary.FileWriter('/home/shenfeng/log',sess.graph) #用于写文件
    plotdata={"batchsize":[],"loss":[]}     #定义一个存放批次值和损失值的字典，存储每批次训练后的loss

    #想模型中输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})     #通过feed_dict机制吧数据灌到占位符

        #生成summery
        summery_str=sess.run(merged_op,feed_dict={X:train_X,Y:train_Y})
        summery_writer.add_summary(summery_str,epoch)
        #显示训练中的详细信息
        if epoch%display_step == 0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})    #获取当前的数据值
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(weight),"B=",sess.run(bais))

            if not (cost=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(weight),"b=",sess.run(bais))
    #保存模型的方法
    #1、
    saver.save(sess, "/home/shenfeng/123456")
    #2、指定存储变量名字与变量的关系
    #tf.train.Saver({'weight':weight,'bais':bais})
    #3、
    #tf.train.Saver([weight,bais])
    #4、将op的名字作为key
    #tf.train.Saver({v.op.name for v in [weight,bais]})



    plt.plot(train_X,train_Y,'ro')
    plt.plot(train_X,sess.run(weight)*train_X+sess.run(bais),label="Fittedline")
    plt.legend()
    plt.show()

    plotdata["avg"]=moving_avg(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avg"],'b--')
    plt.show()














