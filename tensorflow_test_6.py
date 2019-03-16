#-*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
x=np.array([1,2,3])
print(x.shape)
x_1=np.expand_dims(x,axis=-1)
print(x_1.shape)
x_2=np.expand_dims(x,axis=0)
print(x_2.shape)
print(x[2:])
m1=tf.constant([[3,3]])
m2=tf.constant([[3],[3]])
product=tf.matmul(m1,m2)
print(product)
with tf.Session() as sess:
    res=sess.run(product)
    print(res)