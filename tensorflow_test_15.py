'''
tensorboard runtime state
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#namescope

def variable_summary(arg):
    with tf.name_scope("summary"):
        mean=tf.reduce_mean(arg)
        tf.summary.scalar('mean',mean)
        with tf.name_scope("stddev"):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(arg-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(arg))
        tf.summary.scalar('min',tf.reduce_min(arg))
        tf.summary.histogram('histogram',arg)

#load test data
mnist=input_data.read_data_sets("data/minist",one_hot=True)
batch_size=30
n_batch=mnist.train.num_examples//batch_size
#place holder

with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,[None,784],name="x_input")
    y=tf.placeholder(tf.float32,[None,10],name="y_input")
#neural network
with tf.name_scope("layer"):
    with tf.name_scope("weight"):
        w_L1=tf.Variable(tf.zeros([784,10]),name="weight_L1")
        variable_summary(w_L1)
    with tf.name_scope("biasis"):
        b_L1=tf.Variable(tf.zeros([10]),name="biasis_L1")
        variable_summary(b_L1)
    with tf.name_scope("wx_plus_b"):
        xw_plus_b_L1=tf.add(tf.matmul(x,w_L1),b_L1)
    with tf.name_scope("softmax"):
        prediction=tf.nn.softmax(xw_plus_b_L1)

#w_L2=tf.Variable(tf.random.uniform([50,10]),name="weight_L2")
#b_L2=tf.Variable(tf.random.uniform([10]),name="biasis_L2")
#xw_plus_b_L2=tf.add(tf.matmul(out_L1,w_L2),b_L2)
#prediction=tf.nn.softmax(xw_plus_b_L2)

#loss=tf.reduce_mean(tf.square(prediction-y))
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.square(prediction-y))
    tf.summary.scalar('loss',loss)#loss is a single value
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(0.1)
    train=optimizer.minimize(loss)
with tf.name_scope("accuray"):
    with tf.name_scope("predict"):
        predict_value=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(y,axis=1))
    with tf.name_scope("acc"):
        accuracy=tf.reduce_mean(tf.cast(predict_value,tf.float32))
        tf.summary.scalar("accuray",accuracy)
init=tf.global_variables_initializer()
# merge all summary
merge=tf.summary.merge_all()
with tf.Session() as sess:
    writer=tf.summary.FileWriter("log/", sess.graph)
    sess.run(init)

    for epoch in range(22):
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            summary,_=sess.run([merge,train],feed_dict={x:batch_x,y:batch_y})
        writer.add_summary(summary,epoch)
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("epoch:"+str(epoch)+" acc:="+str(acc))


#steps:
#D:\workspace\prototypical-networks-tensorflow>tensorboard --logdir=log
#http://localhost:6006