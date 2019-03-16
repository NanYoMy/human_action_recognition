import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#load test data
mnist=input_data.read_data_sets("data/minist",one_hot=True)
batch_size=80
n_batch=mnist.train.num_examples//batch_size
#place holder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)
#neural network
w1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1),name="weight_L1")
b1=tf.Variable(tf.zeros([500])+0.1,name="biasis_L1")
L1=tf.nn.tanh(tf.matmul(x,w1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

w2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1),name="weight_L2")
b2=tf.Variable(tf.zeros([300])+0.1,name="biasis_L2")
L2=tf.nn.tanh(tf.matmul(L1_drop,w2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

# w3=tf.Variable(tf.random.truncated_normal([100,50],stddev=0.1),name="weight_L3")
# b3=tf.Variable(tf.zeros([50])+0.1,name="biasis_L3")
# L3=tf.nn.tanh(tf.matmul(L2_drop,w3)+b3)
# L3_drop=tf.nn.dropout(L3,keep_prob)

w4=tf.Variable(tf.truncated_normal([300,10],stddev=0.1),name="weight_L4")
b4=tf.Variable(tf.zeros([10])+0.1,name="biasis_L4")
prediction=tf.nn.softmax(tf.matmul(L2_drop,w4)+b4)

# loss=tf.reduce_mean(tf.square(prediction-y))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
optimizer=tf.train.AdamOptimizer(lr)
train=optimizer.minimize(loss)
predict_value=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(predict_value,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(210):
        learn_rate= 0.001 * (0.98**epoch)
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_prob:0.9,lr:learn_rate})
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("epoch:"+str(epoch)+"test_acc:="+str(test_acc)+" train_acc:="+str(train_acc)+" learn rate:"+str(learn_rate))


