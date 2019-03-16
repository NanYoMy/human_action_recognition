import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#load test data
mnist=input_data.read_data_sets("data/mnist",one_hot=True)
batch_size=30
n_batch=mnist.train.num_examples//batch_size
#place holder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#neural network
w_L1=tf.Variable(tf.zeros([784,10]),name="weight_L1")
b_L1=tf.Variable(tf.zeros([10]),name="biasis_L1")
xw_plus_b_L1=tf.add(tf.matmul(x,w_L1),b_L1)
prediction=tf.nn.softmax(xw_plus_b_L1)

#w_L2=tf.Variable(tf.random.uniform([50,10]),name="weight_L2")
#b_L2=tf.Variable(tf.random.uniform([10]),name="biasis_L2")
#xw_plus_b_L2=tf.add(tf.matmul(out_L1,w_L2),b_L2)
#prediction=tf.nn.softmax(xw_plus_b_L2)

#loss=tf.reduce_mean(tf.square(prediction-y))

loss=tf.reduce_mean(tf.square(prediction-y))
optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)
predict_value=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(predict_value,tf.float32))
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("epoch:"+str(epoch)+" acc:="+str(acc))

    saver.save(sess,"model/my_net.ckpt")