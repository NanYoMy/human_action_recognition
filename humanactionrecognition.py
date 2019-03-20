#-*- encoding=utf-8 -*-
#matplotlib inline
from __future__ import print_function
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage import transform,io
import tensorflow as tf
import scipy.io as sio

#train setting
n_epochs = 10
n_episodes = 1500
n_classes=27

n_sample_per_class=32


n_way = n_classes
n_support = 5
n_query = 5

#test setting
n_test_episodes = 1500
n_test_way = n_classes

n_test_support = 5
n_test_query = n_sample_per_class - n_support - n_query#n_test_shot+n_test_query<=22


im_width, im_height, channels = 40, 60, 3
h_dim = 8
z_dim = 16
#tag 92.1


def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=4,padding='SAME')  # 64 filters, each filter will generate a feature map.
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 3)
        return conv
def encoder__conv(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):

        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        #net = conv_block(net, h_dim, name='conv_3')
        #net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)#tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的
        return net

def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder_dilated_conv', reuse=reuse):
        # block_1_in = tf.layers.conv2d(x, h_dim, kernel_size=1, padding='SAME')
        #---------#
        block_1_in=x
        block_1_out = tf.layers.conv2d(block_1_in, h_dim, kernel_size=[2, 3], dilation_rate=[2, 2],padding='SAME')  # 64 filters, each filter will generate a feature map.
        block_1_out = tf.layers.conv2d(block_1_out, h_dim, kernel_size=2,padding='SAME')  # 64 filters, each filter will generate a feature map.
        #---------#

        #---------#
        block_2_in = tf.concat([block_1_out, block_1_in], axis=3)
        block_2_out = tf.layers.conv2d(block_2_in, h_dim, kernel_size=[2, 3], dilation_rate=[4, 4],padding='SAME')
        block_2_out = tf.layers.conv2d(block_2_out, h_dim, kernel_size=2, padding='SAME')
        #---------#

        #---------#
        block_3_in = tf.concat([block_2_out, block_1_out,block_1_in], axis=3)
        block_3_out = tf.layers.conv2d(block_3_in, h_dim, kernel_size=[2, 3], dilation_rate=[6, 6],padding='SAME')
        block_3_out = tf.layers.conv2d(block_3_out, h_dim, kernel_size=3,padding='SAME')  # 64 filters, each filter will generate a feature map.
        #---------#

        #block_3_out = tf.contrib.layers.batch_norm(block_3_out, updates_collections=None, decay=0.99, scale=True, center=True)

        net = tf.layers.conv2d(block_3_out, h_dim, kernel_size=2,padding='SAME')  # 64 filters, each filter will generate a feature map.
        net = tf.nn.relu(net)
        net = tf.contrib.layers.max_pool2d(net, 2)

        net = tf.layers.conv2d(net, h_dim, kernel_size=4,padding='SAME')  # 64 filters, each filter will generate a feature map.
        net = tf.nn.relu(net)
        net = tf.contrib.layers.max_pool2d(net, 2)

        #dense
        net = tf.contrib.layers.flatten(net)#tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的

        return net

def euclidean_distance(a, b): # a是query b是protypical
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

def load_data(path):
    data = sio.loadmat(path)
    skelet=data['d_skel']
    #print(skelet.shape)#(20,3,48) 20个点，每个点有xyz, 一共48帧，这里需要处理不帧数的样本
    return skelet
def Normalize(data,factor):
    m = np.mean(data)
    mx = data.max()
    mn = data.min()
    return (data - mn) / factor

def max_diff_channal(feature):
    diff=np.zeros([3])
    for i in range(feature.shape[2]):
        diff[i]=feature[:,:,i].max()-feature[:,:,i].min()
    return diff.max()
# 使用其余点减去中心点的距离
def resize(diff_feature):
    sample=np.zeros([im_width,im_height,3])
    sample[:, :, 0] = transform.resize(diff_feature[:, :, 0], (im_width, im_height), mode='reflect', anti_aliasing=True)
    sample[:, :, 1] = transform.resize(diff_feature[:, :, 1], (im_width, im_height), mode='reflect', anti_aliasing=True)
    sample[:, :, 2] = transform.resize(diff_feature[:, :, 2], (im_width, im_height), mode='reflect', anti_aliasing=True)
    return sample


# 使用其余点减去中心点的距离
def get_diff_feature(skelet,ref_point_index=3):#第三个点刚刚好是hip center
    feature=skelet.swapaxes(1,2)
    for i in range(feature.shape[1]):
        feature[:,i,:]=feature[:,i,:]-np.repeat(np.expand_dims(feature[ref_point_index, i, :], axis=0),feature.shape[0],axis=0)
    im=np.delete(feature,2,axis=0)
    factor=max_diff_channal(im)
    for i in range(im.shape[2]):
        im[:,:,i]=Normalize(im[:,:,i],factor)
    sample=resize(im)
    return sample

def getall(data_addr,n_classes,offset=0):
    data_set=np.zeros([n_classes,n_sample_per_class,im_height, im_width,3], dtype=np.float32)
    for j in range(len(data_addr)):
        skelet = load_data(data_addr[j])# skelet是numpy的ndarray类型
        token = data_addr[j].split('\\')[-1].split('_')
        i=int(token[0][1:])-1-offset#class
        j=(int(token[1][1:])-1)*4+int(token[2][1:])-1#id
        sample=get_diff_feature(skelet)
        data_set[i,j]=sample.swapaxes(1,0)
    return data_set

def prepar_data(data_addr,n_classes):
    train_data_set = np.zeros([n_classes, n_query + n_support, im_height, im_width, 3], dtype=np.float32)
    test_data_set = np.zeros([n_classes, n_sample_per_class - n_query - n_support, im_height, im_width, 3], dtype=np.float32)
    all_data_set=getall(data_addr, n_classes)
    selected=np.random.permutation(n_sample_per_class)
    train_sample_id = selected[:n_query + n_support]

    test_sample_id = selected[n_query + n_support:]

    for i in range(n_classes):
        for j, id in enumerate(train_sample_id):
            train_data_set[i,j]=all_data_set[i,id]#n_query+n_shot training sample
    for i in range(n_classes):
        for j, id in enumerate(test_sample_id):
            test_data_set[i,j]=all_data_set[i,id]#n_query+n_shot training sample
    return test_data_set,train_data_set


data_addr = sorted(glob.glob('.\\data\\Skeleton\\data\\*.mat'))# all data
test_dataset,train_dataset=prepar_data(data_addr, n_classes)
print(train_dataset.shape)#(10, 32, 60, 40, 3)
print(test_dataset.shape)#(10, 32, 60, 40, 3)
regularizer = tf.contrib.layers.l1_regularizer(0.0)

x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])

x_shape = tf.shape(x)
q_shape = tf.shape(q)
#训练的时候具有support sample的参数
num_classes, num_support = x_shape[0], x_shape[1]# num_class num_support_sample
num_queries = q_shape[1]#num_query_sample
#y为label数据由外部导入
y = tf.placeholder(tf.int64, [None, None])
y_one_hot = tf.one_hot(y, depth=num_classes)# dimesion of each one_hot vector
#emb_x是样本通过encoder之后的结果
emb_x = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
emb_dim = tf.shape(emb_x)[-1] # the last dimesion

emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)#计算每一类的均值，每一个类的样本都通过CNN映射到高维度空间
emb_q = encoder(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)

dists = euclidean_distance(emb_q, emb_x)

log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])#-1表示自动计算剩余维度，paper中公式2
ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))#reshpae(a,[-1])会展开所有维度, ce_loss=cross entropy
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))
# regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
train_op = tf.train.AdamOptimizer().minimize(ce_loss)
sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
# Test_A=tf.placeholder(tf.int64, [None, 40,60,None])
# Test_B=tf.placeholder(tf.int64, [None, 40,60,None])
# Test_C=tf.concat([Test_A,Test_B],axis=[0,3])
sess.run(init_op)
# avg_acc = 0.
# avg_ls=0.
# print('average acc: {:.5f}, average loss: {:.5f}'.format(avg_acc,avg_ls ))
for epi in range(n_episodes):
    '''
    随机产生一个数组，包含0-n_classes,取期中n_way个类
    '''
    epi_classes = np.random.permutation(n_classes)[:n_way]  # n_way表示类别
    support = np.zeros([n_way, n_support, im_height, im_width, channels], dtype=np.float32)  # n_shot表示样本的数目
    query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        '''
        选n_shot+n_query进行训练
        n_shot是作为参数，n_query作为训练样本
        '''
        #selected = np.random.permutation(n_sample_per_class)[:n_shot + n_query]
        #only 10 sample will used to train the model
        selected = np.random.permutation(n_support + n_query)[:n_support + n_query]
        support[i] = train_dataset[epi_cls, selected[:n_support]]
        query[i] = train_dataset[epi_cls, selected[n_support:]]
    # support = np.expand_dims(support, axis=-1)
    # query = np.expand_dims(query, axis=-1)
    labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
    # c=sess.run(Test_C,feed_dict={Test_A:np.zeros([30,40,60,3]),Test_B:np.ones([40,40,60,1])})
    # print(c.shape)
    _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y: labels})
    # avg_acc += ac
    # avg_ls += ls
    if (epi + 1) %50 == 0:
        print('[ episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_episodes,ls, ac))
    # if ls<0.1 :
    #     print('[ episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_episodes, ls, ac))
    #     break
# Load Test Dataset
# avg_acc /= n_episodes
# avg_ls/=n_episodes
# print('average acc: {:.5f}, average loss: {:.5f}'.format(avg_acc,avg_ls ))


print('Testing normal classes...')
avg_acc = 0.
avg_ls=0.
for epi in range(n_test_episodes):
    epi_classes = np.random.permutation(n_classes)[:n_test_way]
    support = np.zeros([n_test_way, n_test_support, im_height, im_width, channels], dtype=np.float32)
    query = np.zeros([n_test_way, n_test_query, im_height, im_width,channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):

        selected_support = np.random.permutation(n_query+n_support)[:n_support]#从训练集合取support样本
        selected_query = np.random.permutation(n_test_query)#22个样本
        support[i] = train_dataset[epi_cls, selected_support]#从训练集合取support样本
        query[i] = test_dataset[epi_cls, selected_query]
    # support = np.expand_dims(support, axis=-1)
    # query = np.expand_dims(query, axis=-1)
    labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
    ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y:labels})
    avg_acc += ac
    avg_ls+=ls
    if (epi+1) % 50 == 0:
        print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))
avg_acc /= n_test_episodes
avg_ls/=n_test_episodes
print('Average Test Accuracy: {:.5f} Average loss : {:.5f}'.format(avg_acc,avg_ls))

'''
#， 训练样本的修改：现在每次的都是从32个样本中随机抽取5个作为支持向量，5个作为query向量。能否改成只有在10个样本中进行随机抽取， 完成
#， 测试的修改：分27类 现在每次测试都是从32个样本里面随机抽5个当作支持向量，检验剩余27样本的数据，
图片的生成：现在每次都是从sequnce中生成一张图片，能否生成多张图片？
'''