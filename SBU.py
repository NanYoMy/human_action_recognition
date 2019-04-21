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
'''
training:使用4个support样本，利用4个query,对模型进行训练
inference:使用4个从train样本中得到的support样本，对剩余的24样本进行评估，
'''
n_joint=15
n_episodes = 60
n_classes=8
n_sample_per_class=15
n_way = 8
n_support = 4
n_query = 4
#test setting
n_test_episodes = 1000
n_test_way = n_way
n_test_support = n_support
n_test_query = n_sample_per_class - n_support - n_query#n_test_shot+n_test_query<=22

im_height,im_width,  channels = 15,50,6
h_dim = 8
z_dim = 64

ckpt_path='./ckpt/untitled'
def euclidean_distance(query=None, prototype=None): # a是query b是prototype
    N, D = tf.shape(query)[0], tf.shape(query)[1]
    M = tf.shape(prototype)[0]
    query = tf.tile(tf.expand_dims(query, axis=1), (1, M, 1))
    prototype = tf.tile(tf.expand_dims(prototype, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(query - prototype), axis=2)

def load_txt_data(path):
    skelet = np.loadtxt(path, delimiter=",", dtype=np.float32)
    skelet=np.delete(skelet,0,axis=1)
    skeletA,skeletB=np.split(skelet,[45],1)
    frame=skeletA.shape[0]
    skeletA=skeletA.reshape(frame,n_joint,3)
    frame=skeletB.shape[0]
    skeletB=skeletB.reshape(frame,n_joint,3)
    skeletA=skeletA.swapaxes(1,0)
    skeletB = skeletB.swapaxes(1, 0)
    # skelet[:,:,0]=skelet[:,:,0]-skelet[:,:,0].mean()
    # skelet[:, :, 1] = skelet[:, :, 1] - skelet[:, :,1].mean()
    # skelet[:, :, 2] = skelet[:, :, 2] - skelet[:, :, 2].mean()
    return skeletA,skeletB
def Normalize(data,factor):
    m = np.mean(data)
    mx = data.max()
    mn = data.min()
    return (data - mn) / factor
def get_channal_max_range(skelet):
    diff=np.zeros([3])
    for i in range(skelet.shape[2]):
        diff[i]=skelet[:,:,i].max()-skelet[:,:,i].min()
    return diff.max()
# 使用其余点减去中心点的距离
def resize(feature):
    sample=np.zeros([im_height,im_width,channels])
    for i in range(channels):
        sample[:, :, i] = transform.resize(feature[:, :, i], (im_height, im_width), mode='reflect', anti_aliasing=True)
    return sample
# 使用其余点减去中心点的距离
cur_action=0
def nextBatch(test_set):
    global cur_action
    sample_per_action=test_set[cur_action]
    query = np.zeros([1, len(sample_per_action), im_height, im_width, channels], dtype=np.float32)
    query[0]=sample_per_action
    return cur_action,len(sample_per_action),query
def nextBatch2(test_set):
    global cur_action
    sample_per_action=test_set[cur_action]
    query = np.zeros([1, sample_per_action.shape[0], im_height, im_width, channels], dtype=np.float32)
    query[0]=sample_per_action
    return cur_action,sample_per_action.shape[0],query

def data_fix(feature,ref_point_index=3):

    for i in range(feature.shape[1]):
        feature[:,i,:]=feature[:,i,:]-np.repeat(np.expand_dims(feature[ref_point_index, i, :], axis=0),feature.shape[0],axis=0)
    feature_new=np.delete(feature,ref_point_index,axis=0)
    return feature_new

def normalize_skeleton(skeletA, skeletB):#第三个点刚刚好是hip center
    # skeletB=data_fix(skeletB,2)
    # skeletA = data_fix(skeletA,2 )
    factor=get_channal_max_range(skeletA)
    for i in range(skeletA.shape[2]):
        skeletA[:,:,i]=Normalize(skeletA[:,:,i],factor)
    factor = get_channal_max_range(skeletB)
    for i in range(skeletB.shape[2]):
        skeletB[:,:,i]=Normalize(skeletB[:,:,i],factor)

    return skeletA,skeletB
def ouput_3_gray_imge(diff_feature,path):
    prename = path.split('\\')[-1]
    print(prename)
    x_im=diff_feature[:, :, 0]*255
    im = Image.fromarray(x_im.astype(np.uint8))
    im.save((".\\data\\Skeleton5\\img\\x_%s.bmp") % (prename))

    y_im=diff_feature[:, :, 1]*255
    im = Image.fromarray(y_im.astype(np.uint8))
    im.save((".\\data\\Skeleton5\\img\\y_%s.bmp") % (prename))

    z_im=diff_feature[:, :, 2]*255
    im = Image.fromarray(z_im.astype(np.uint8))
    im.save((".\\data\\Skeleton5\\img\\z_%s.bmp") % (prename))
# def prepar_data(data_addr,n_classes,offset=0):
#     train_set=np.zeros([n_classes,n_query+n_support,im_height, im_width,channels], dtype=np.float32)
#     test_set={}
#     for i in range(n_classes):
#         test_set[i]=[]
#     index=np.zeros([8],np.int)
#     for addr in data_addr:
#         skeletA,skeletB = load_txt_data(addr)# skelet是numpy的ndarray类型
#         token = addr.split('\\')[-1].split('-')
#         i=int(token[0])-1#class
#         j=index[i]
#         sampleA, sampleB = normalize_skeleton(skeletA, skeletB)
#         # merge_sample=resize(np.concatenate((sampleA, sampleB), axis=2))
#         merge_sample = resize(np.vstack((sampleA, sampleB)))
#         if(index[i]<(n_query+n_support)):#小于8的
#             train_set[i, j] = resize(merge_sample)
#             index[i] = index[i] + 1
#         else:
#            test_set[i].append(merge_sample)
#
#     return train_set,test_set
def prepar_data_matrix(data_addr,n_classes):
    train_data_set = np.zeros([n_classes, n_query + n_support, im_height, im_width,channels], dtype=np.float32)
    test_data_set = np.zeros([n_classes, n_sample_per_class - n_query - n_support, im_height, im_width, channels], dtype=np.float32)
    train_index = np.zeros([8], np.int)
    test_index = np.zeros([8], np.int)
    for addr in data_addr:
        skeletA, skeletB = load_txt_data(addr)  # skelet是numpy的ndarray类型
        token = addr.split('\\')[-1].split('-')
        i = int(token[0]) - 1  # class
        sampleA, sampleB = normalize_skeleton(skeletA, skeletB)
        # ouput_3_gray_imge(sampleA,addr+'_a')
        # ouput_3_gray_imge(sampleB, addr + '_b')
        merge_sample = resize(np.concatenate((sampleA, sampleB), axis=2))
        # merge_sample = resize(np.vstack((sampleA, sampleB)))
        if (train_index[i] < (n_query + n_support)):  # 小于8的
            j = train_index[i]
            train_data_set[i, j] = resize(merge_sample)
            train_index[i] = train_index[i] + 1
        elif (test_index[i]<(n_sample_per_class-n_query-n_support)):
            j = test_index[i]
            test_data_set[i, j]= resize(merge_sample)
            test_index[i]=test_index[i]+1
        else:
            pass
    return train_data_set,test_data_set

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
def train_test():
    data_addr = sorted(glob.glob('.\\data\\Skeleton5\\*.txt'))# all data
    train_dataset,test_dataset=prepar_data_matrix(data_addr, n_classes)
    print(train_dataset.shape)
    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels],name="x")
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels],name="q")
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    #训练的时候具有support sample的参数
    num_classes, num_support = x_shape[0], x_shape[1]# num_class num_support_sample
    numb_queris_class,num_queries = q_shape[0], q_shape[1]#num_query_sample
    #y为label数据由外部导入
    y = tf.placeholder(tf.int64, [None, None],name="y")
    y_one_hot = tf.one_hot(y, depth=num_classes)# dimesion of each one_hot vector
    #emb_x是样本通过encoder之后的结果
    emb_x = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim,reuse=False)
    emb_dim = tf.shape(emb_x)[-1] # the last dimesion
    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)#计算每一类的均值，每一个类的样本都通过CNN映射到高维度空间

    emb_q = encoder(tf.reshape(q, [numb_queris_class * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)

    dists = euclidean_distance(emb_q, emb_x)

    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [numb_queris_class, num_queries, -1])#-1表示自动计算剩余维度，paper中公式2 log_softmax 默认 axis=-1
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]),name='loss')#reshpae(a,[-1])会展开所有维度, ce_loss=cross entropy
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)),name='acc')

    tf.add_to_collection('acc', acc)
    tf.add_to_collection('loss', ce_loss)

    train_op = tf.train.AdamOptimizer().minimize(ce_loss)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for epi in range(n_episodes):
        '''
        随机产生一个数组，包含0-n_classes,取期中n_way个类
        '''

        epi_classes = np.random.permutation(n_classes)[:n_way]
        support = np.zeros([n_way, n_support, im_height, im_width, channels], dtype=np.float32)  # n_shot表示样本的数目
        query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_support + n_query)[:n_support + n_query]
            support[i] = train_dataset[epi_cls, selected[:n_support]]
            query[i] = train_dataset[epi_cls, selected[n_support:]]
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y: labels})
        #if (epi + 1) %50 == 0:
        print('[ episode {}/{}] => loss: {:.5f}, acc: {:.5f} '.format(epi + 1,n_episodes,ls,ac))

    saver.save(sess, ckpt_path)
    print('Testing normal classes...')
    avg_acc = 0.
    avg_ls=0.

    for epi in range(n_test_episodes):
        epi_classes = np.arange(n_classes)[:n_test_way]
        # epi_classes=np.arange(n_test_way)[:n_test_way]
        support = np.zeros([n_test_way, n_test_support, im_height, im_width, channels], dtype=np.float32)
        query = np.zeros([n_test_way,n_test_query,im_height,im_width,channels],dtype=np.float32)

        for i, epi_cls in enumerate(epi_classes):
            selected_support = np.arange(n_query+n_support)[:n_test_support]#从训练集合取support样本
            support[i] = train_dataset[epi_cls, selected_support]#从训练集合取support样本
            selected_query = np.arange(n_test_query)
            query[i] = train_dataset[epi_cls, selected_query]

        labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)

        ls_1, ac_1, log_p_1, distance_1, out_emb_q_1, out_emb_x_1 = sess.run([ce_loss, acc, log_p_y, dists, emb_q, emb_x],feed_dict={x: support, q: query, y: labels})
        avg_acc += ac_1
        avg_ls += ls_1
        if (epi + 1) % 50 == 0:
            print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f} '.format(epi + 1, n_test_episodes, ls_1, ac_1))
    # sess.close()

    print('Testing normal classes...')
    avg_acc = 0.
    avg_ls=0.

    for epi in range(n_test_episodes):
        epi_classes = np.arange(n_classes)[:n_test_way]
        # epi_classes=np.arange(n_test_way)[:n_test_way]
        support = np.zeros([n_test_way, n_test_support, im_height, im_width, channels], dtype=np.float32)
        query2 = np.zeros([1,n_test_query,im_height,im_width,channels],dtype=np.float32)

        for i, epi_cls in enumerate(epi_classes):
            selected_support = np.arange(n_query+n_support)[:n_test_support]#从训练集合取support样本
            support[i] = train_dataset[epi_cls, selected_support]#从训练集合取support样本

        selected_query = np.arange(n_test_query)
        test_tmp_aciton=0
        query2[0] = train_dataset[test_tmp_aciton, selected_query]
        for i in epi_classes:
            if epi_classes[i]==test_tmp_aciton:
                test_tmp_aciton=i
                break
        labels = np.tile(np.array([test_tmp_aciton])[:, np.newaxis], (1, n_test_query)).astype(np.uint8)

        ls, ac,log_p,distance,out_emb_q,out_emb_x = sess.run([ce_loss, acc,log_p_y,dists,emb_q,emb_x], feed_dict={x: support, q: query2, y: labels})
        avg_acc += ac
        avg_ls += ls
        if (epi + 1) % 50 == 0:
            print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f} '.format(epi + 1, n_test_episodes, ls, ac))
    sess.close()

