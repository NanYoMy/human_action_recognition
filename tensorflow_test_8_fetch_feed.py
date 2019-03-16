import tensorflow as tf
#fetch
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
add=tf.add(input1,input2)
mul=tf.multiply(input3,add)
with tf.Session() as sess:
    res=sess.run([mul,add])
    print(res)
#feed
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
res=tf.add(input1,input2)
with tf.Session() as sess:
    print(sess.run(res,feed_dict={input1:[7.0],input2:[2.0]}))