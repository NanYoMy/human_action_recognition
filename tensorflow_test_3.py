import tensorflow as tf

x = tf.constant([[1., 1.],
                 [2., 2.]])
tf.reduce_mean(x)  # 1.5
m1 = tf.reduce_mean(x, axis=0)  # [1.5, 1.5]
m2 = tf.reduce_mean(x, 1)  # [1.,  2.]

xx = tf.constant([[[1., 1, 1],
                   [2., 2, 2]],

                  [[3, 3, 3],
                   [4, 4, 4]]])
m3 = tf.reduce_mean(xx, [0, 1]) # [2.5 2.5 2.5]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(m1))
    print(sess.run(m2))

    print(xx.get_shape())
    print(sess.run(m3))

