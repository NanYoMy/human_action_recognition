import tensorflow as tf

test=tf.constant("hello")

with tf.Session() as sess:
    sess.run(test)
print("finish")