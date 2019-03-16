
import tensorflow as tf

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])

with tf.Session() as sess:
    a_shape = tf.shape(a)
    a_getshape = a.get_shape()
    print(a_shape[-1])
    print(a_getshape)
