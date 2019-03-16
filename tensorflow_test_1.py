import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
const=tf.constant(2.0,name='const')
b=tf.Variable(2.0,name='b')
c=tf.Variable(3.0,name='c')
a=tf.multiply(tf.add(b,c),tf.add(const,c),name='a')
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    a_out=sess.run(a)
    print("varable is {}".format(a_out))

