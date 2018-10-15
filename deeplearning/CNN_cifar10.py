#coding=utf-8
import tensorflow as tf
import cifar10,cifar10_input
import numpy as np
import time

batchsize=300
dir="/home/shenfeng/PycharmProjects/cifar-10-batches-bin"
#cifar10.maybe_download_and_extract()
images_train,labels_train=cifar10_input.distorted_inputs(data_dir=dir,batch_size=batchsize)
images_test,lables_test=cifar10_input.inputs(eval_data=True,data_dir=dir,batch_size=batchsize)

def variable_with_loss(shape,wl,stddev):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

input_images=tf.placeholder(tf.float32,[batchsize,24,24,3])
input_labels=tf.placeholder(tf.int32,[batchsize])

w1=variable_with_loss([5,5,3,64],wl=0.0,stddev=0.05)
b1=tf.Variable(tf.constant(0.0,shape=[64]))
covn1=tf.nn.conv2d(input_images,w1,[1,1,1,1],'SAME')
covn1_relu=tf.nn.relu(tf.nn.bias_add(covn1,b1))
maxpool1=tf.nn.max_pool(covn1_relu,[1,3,3,1],[1,2,2,1],'SAME')
layer1=tf.nn.lrn(maxpool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

w2=variable_with_loss([5,5,64,64],wl=0.0,stddev=0.05)
b2=tf.Variable(tf.constant(0.1,shape=[64]))
covn2=tf.nn.conv2d(layer1,w2,[1,1,1,1],'SAME')
covn2_relu=tf.nn.relu(tf.nn.bias_add(covn2,b2))
layer2=tf.nn.lrn(covn2_relu,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
maxpool1=tf.nn.max_pool(layer2,[1,3,3,1],[1,2,2,1],'SAME')

#full_connected_1
in1=tf.reshape(maxpool1,[batchsize,-1])
in_dim=in1.get_shape()[1].value
w_full1=variable_with_loss([in_dim,384],wl=0.04,stddev=0.004)
bias_full=tf.Variable(tf.constant(0.1,shape=[384]))
full_1=tf.nn.relu(tf.matmul(in1,w_full1)+bias_full)


#full_connected2
w_full2=variable_with_loss([384,192],wl=0.04,stddev=0.004)
bias_full2=tf.Variable(tf.constant(0.1,shape=[192]))
full_2=tf.nn.relu(tf.matmul(full_1,w_full2)+bias_full2)

#last_out
w_last=variable_with_loss([192,10],wl=0.0,stddev=1.0/192)
b_last=tf.Variable(tf.constant(0.0,shape=[10]))
out_last=tf.matmul(full_2,w_last)+b_last

#LOSS
def loss(labels,logits):
    labels=tf.cast(labels,tf.int64)
    cross_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits),name="cross_entorpy_mean")
    tf.add_to_collection('losses',cross_loss)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

Loss=loss(labels=input_labels,logits=out_last)

train=tf.train.AdamOptimizer(0.001).minimize(Loss)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
max_step=30000
top_k_op=tf.nn.in_top_k(out_last,tf.cast(input_labels,tf.int64),1)
for step in range(max_step):
    start_time=time.time()
    imahes_batch,labels_batch=sess.run([images_train,labels_train])
    _,loss_value=sess.run([train,Loss],feed_dict={input_images:imahes_batch,input_labels:labels_batch})
    during=time.time()-start_time

    if step % 10==0:
        per_sec=batchsize/during
        sec_per_batch=float(during)
        print("step %d  Loss= %.2f   examples_per_sec= %.1f  time_per_batch= %.3f"%(step,loss_value,per_sec,sec_per_batch))

import math
num_examples=10000
num_iter=int(math.ceil(num_examples/batchsize))
true_count=0
total_sample_count=num_iter*batchsize
step=0
while step<num_iter:
    imagesbatch,labelsbatch=sess.run([images_test,lables_test])
    predictions=sess.run([top_k_op],feed_dict={input_images:imagesbatch,input_labels:labelsbatch})

    true_count+=np.sum(predictions)
    step+=1

print("precision=%.3f"%(true_count*1.0/total_sample_count))