import tensorflow as tf
import numpy as np
const=tf.constant(2.0,name='const')
b=tf.placeholder(tf.float32,name='b')
c=tf.Variable(3.0,name='c')
a=tf.multiply(tf.add(b,c),tf.add(const,c),name='a')
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    a_out=sess.run(a,feed_dict={b: np.arange(0, 10)})
print("varaible a is{}".format(a_out))
b=tf.placeholder(tf.float32,[None,1],name='b')
a=tf.multiply(tf.add(b,c),tf.add(const,c),name='a')
with tf.Session() as sess:
    sess.run(init_op)
    a_out=sess.run(a,feed_dict={b: np.arange(0, 10)[:,np.newaxis]})
print("varaible a is{}".format(a_out))