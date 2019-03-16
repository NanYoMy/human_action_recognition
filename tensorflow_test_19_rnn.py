# -*- encoding=utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print(tf.VERSION)
mnist=input_data.read_data_sets("data/mnist",one_hot=True)
n_inputs=28
max_time=28
lstm_size=100
n_classes=10
batch_size=50
n_batch=mnist.train.num_examples//batch_size
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
biasis=tf.Variable(tf.constant(0.1,shape=[n_classes]))
def RNN(x,weights,biasis):
    x_image=tf.reshape(x,[-1,n_inputs,max_time])
    lstm_cell= tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # final=[state,batch_size,cell.stat_size(ltsm_size:100)]
    #final[0] cell state  ; final[1] hidden state
    # outputs=[batch_size,max_time,output_size(100)]
    outputs,final=tf.nn.dynamic_rnn(lstm_cell,x_image,dtype=tf.float32)
    result=tf.nn.softmax(tf.matmul(final[1],weights)+biasis)
    return result

prediction=RNN(x,weights,biasis)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for _ in range(n_batch):
            batch_x,bathc_y=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:bathc_y})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("%d acc=%f"%(_,acc))