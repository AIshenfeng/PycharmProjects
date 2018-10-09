import tensorflow as tf

global_step=tf.Variable(0,trainable=False)
start_learn_rate=1.0
learn_rate=tf.train.exponential_decay(start_learn_rate,global_step,10,0.9)

add=global_step.assign_add(1)
opt=tf.train.GradientDescentOptimizer(learn_rate)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):

        a,rate=sess.run([add, learn_rate])
        print(a,rate)

