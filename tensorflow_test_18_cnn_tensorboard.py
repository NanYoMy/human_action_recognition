#-*- encoding=utf8 -*-
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

mnist=input_data.read_data_sets("data/mnist",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size

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



#generator weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasis_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2D(x,W):
    '''
    :param x: [batch,height,width,channal]
    :param W: filter [filter_height,filter_width,input_channal,output_channal]
    striders:strides[1]x方向步长    strides[2]y方向上步长
    :return:
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    '''
    ksize[1]x轴方向 ksize[2]y轴方向; stride[1]x轴方向  strider[2]y轴方向
    :param x:
    :return:
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,shape=[None,784],name="input_x")
    y=tf.placeholder(tf.float32,shape=[None,10],name="input_y")
with tf.name_scope("input_image"):
    x_image=tf.reshape(x,[-1,28,28,1])

with tf.name_scope("conv1"):
    w_conv1=weight_variable([5,5,1,32])
    b_conv1=biasis_variable([32])

    h_conv1=tf.nn.relu(conv2D(x_image,w_conv1)+b_conv1)#激活函数
    h_pool1=max_pool_2x2(h_conv1)
with tf.name_scope("conv2"):
    w_conv2=weight_variable([5,5,32,64])
    b_conv2=biasis_variable([64])
    h_conv2=tf.nn.relu(conv2D(h_pool1, w_conv2) + b_conv2)
    h_pool2=max_pool_2x2(h_conv2)
with tf.name_scope("fc1"):
    w_fc1=weight_variable([7*7*64,1024])
    b_fc1=biasis_variable([1024])

    h_pool2_flatten=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flatten,w_fc1)+b_fc1)

    keep_pro=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_pro)

with tf.name_scope("fc2"):
    w_fc2=weight_variable([1024,10])
    b_fc2=biasis_variable([10])
    h_fc2=tf.matmul(h_fc1_drop,w_fc2)+b_fc2
    prediction=tf.nn.softmax(h_fc2)

with tf.name_scope("train"):
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    train=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    tf.summary.scalar("loss",cross_entropy)
with tf.name_scope("accuray"):
    correct_prediction=tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuray=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accurary", accuray)
init=tf.global_variables_initializer()
merge=tf.summary.merge_all()
with tf.Session() as sess:
    writer_test=tf.summary.FileWriter("log/train",graph=sess.graph)
    writer_train = tf.summary.FileWriter("log/test", graph=sess.graph)
    sess.run(init)
    for epoch in range(2):
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_pro:0.7})

            summary=sess.run(merge,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_pro:1.0})
            writer_test.add_summary(summary,batch)


        acc=sess.run(accuray,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_pro:1.0})
        print("epoch:"+str(epoch)+" accuray:"+str(acc))




