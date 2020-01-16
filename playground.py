import tensorflow as tf
X = tf.constant([[1,2,3], [3,2,4]], dtype=tf.float32)
W = tf.constant([[1,1],[2,2],[3,3]], dtype=tf.float32)
bias = tf.constant([1, 2], dtype=tf.float32)
y = tf.nn.softmax(tf.matmul(X, W) + bias)
with tf.Session() as sess:
    print(sess.run(tf.matmul(X, W) + bias))
    print(sess.run(y))
    """
    [[0.26894143 0.7310586 ]
    [0.26894143 0.7310586 ]]
    """
import tensorflow as tf
A = [[1.0,3.0],[1.0,3.0]]
print(A)
with tf.Session() as sess:
        print(sess.run(tf.nn.softmax(A)))