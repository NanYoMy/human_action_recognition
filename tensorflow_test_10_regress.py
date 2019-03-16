#-*- encoding=utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#sample
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis] #扩展成二维的数据
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
#place holder, None可以是随机数目的样本
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#neural network
weight_L1=tf.Variable(tf.random.normal([1,10]))#输入层一个神经元，中间层10个神经元
biasis_L1=tf.Variable(tf.zeros([1,10]))
wx_plus_b_L1=tf.matmul(x,weight_L1)+biasis_L1
out_L1=tf.nn.tanh(wx_plus_b_L1)

weight_L2=tf.Variable(tf.random.normal([10,1]))
biasis_L2=tf.Variable(tf.zeros([1,1]))
wx_plus_b_L2=tf.matmul(out_L1,weight_L2)+biasis_L2#matmul左右位置不能交换
prediction=tf.nn.tanh(wx_plus_b_L2)

loss=tf.reduce_mean(tf.square(y-prediction))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    pred_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_value,'r-',lw=5)
    plt.show()