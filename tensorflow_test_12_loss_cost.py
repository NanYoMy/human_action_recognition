import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#load test data
mnist=input_data.read_data_sets("data/minist",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size
#place holder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#neural network
w_L1=tf.Variable(tf.zeros([784,10]),name="weight_L1")
b_L1=tf.Variable(tf.zeros([10]),name="biasis_L1")
xw_plus_b_L1=tf.add(tf.matmul(x,w_L1),b_L1)
prediction=tf.nn.softmax(xw_plus_b_L1)
# loss=tf.reduce_mean(tf.square(prediction-y))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)
predict_value=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(predict_value,tf.float32))
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2100):
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("epoch:"+str(epoch)+" acc:="+str(acc))

