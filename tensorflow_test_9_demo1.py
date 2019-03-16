#-*- encoding=utf8 -*-
import tensorflow as tf
import numpy as np
#sample
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2
#model
b=tf.Variable(1.)
k=tf.Variable(1.5)
y=k*x_data+b
#loss 
loss=tf.reduce_mean(tf.square(y_data-y))
#定义GD
optimizer=tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))