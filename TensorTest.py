import  tensorflow as tf
import numpy as np

def encoder(x, h_dim, z_dim,reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):#reuse非常有用，可以避免设置
        # block_1_in = tf.layers.conv2d(x, h_dim, kernel_size=1, padding='SAME')
        #---------#

        block_1_in=tf.layers.conv2d(x, h_dim, kernel_size=[2, 3], dilation_rate=[2, 2],padding='SAME')
        block_1_out = tf.layers.conv2d(block_1_in, h_dim, kernel_size=[2, 3], dilation_rate=[2, 2],padding='SAME')  # 64 filters, each filter will generate a feature map.
        # block_1_out = tf.contrib.layers.batch_norm(block_1_out, updates_collections=None, decay=0.99, scale=True, center=True)
        block_1_out = tf.nn.relu(block_1_out)
        #---------#

        #---------#
        block_2_in = tf.concat([block_1_out, block_1_in], axis=3)
        block_2_out = tf.layers.conv2d(block_2_in, h_dim*2, kernel_size=[2, 3], dilation_rate=[2, 4],padding='SAME')
        # block_2_out = tf.contrib.layers.batch_norm(block_2_out, updates_collections=None, decay=0.99, scale=True,center=True)
        block_2_out = tf.nn.relu(block_2_out)
        #---------#

        #---------#
        block_3_in = tf.concat([block_2_out, block_1_out,block_1_in], axis=3)
        block_3_out = tf.layers.conv2d(block_3_in, h_dim*3, kernel_size=[2, 3], dilation_rate=[2, 4],padding='SAME')
        # block_3_out = tf.contrib.layers.batch_norm(block_3_out, updates_collections=None, decay=0.99, scale=True,center=True)
        block_3_out = tf.nn.relu(block_3_out)
        #---------#
        # ---------#
        net = tf.concat([block_3_out,block_2_out, block_1_out, block_1_in], axis=3)
        # block_4_out = tf.layers.conv2d(block_4_in, h_dim, kernel_size=[2, 3], dilation_rate=[2, 2], padding='SAME')
        # # block_3_out = tf.contrib.layers.batch_norm(block_3_out, updates_collections=None, decay=0.99, scale=True,center=True)
        # block_4_out = tf.nn.relu(block_4_out)
        # ---------#

        net = tf.layers.conv2d(net, h_dim*8, kernel_size=5,padding='SAME')  # 64 filters, each filter will generate a feature map.
        # net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [1,2],strides=[1, 2])
        net = tf.layers.conv2d(net, h_dim*4, kernel_size=5,padding='SAME')  # 64 filters, each filter will generate a feature map.
        # net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [2, 3], strides=[2, 3])
        net = tf.layers.conv2d(net, h_dim*2, kernel_size=5,padding='SAME')  # 64 filters, each filter will generate a feature map.
        # net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [2, 3], strides=[2, 3])
        #dense
        net = tf.layers.flatten(net)#tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
        return net
x=tf.placeholder(tf.float32,[None,15,50,3])
y=encoder(x,2,1,reuse=False)
sess=tf.InteractiveSession()
init_op=tf.global_variables_initializer()
sess.run(init_op)
x1=np.random.rand(2,15,50,3)
x2=np.random.rand(2,15,50,3)
x3=np.random.rand(2,15,50,3)
all=np.vstack((x1,x2))
y1=sess.run([y],feed_dict={x:x1})
y2=sess.run([y],feed_dict={x:all})
print(y1)
print(y2)

