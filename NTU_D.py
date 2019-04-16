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
import shutil
#train setting
'''
training:使用4个support样本，利用4个query,对模型进行训练
inference:使用4个从train样本中得到的support样本，对剩余的24样本进行评估，
'''
n_epochs = 20
n_episodes = 300
n_classes=60
n_way = n_classes
n_support = 5
n_query = 5
#test setting
n_test_episodes = 1000
n_test_way = n_classes
n_test_support = n_support

im_height,im_width,  channels = 25, 90, 3
h_dim = 8
z_dim = 64
ckpt_path='./ckpt/untitled'
def euclidean_distance(query=None, prototype=None): # a是query b是protypical
    # a.shape = Class_Number*Query x D
    # b.shape = Class_Number x D
    N, D = tf.shape(query)[0], tf.shape(query)[1]
    M = tf.shape(prototype)[0]
    query = tf.tile(tf.expand_dims(query, axis=1), (1, M, 1))
    prototype = tf.tile(tf.expand_dims(prototype, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(query - prototype), axis=2)
def load_data(path):
    data = sio.loadmat(path)
    skelet=data['rgb']

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
Dir="D:\\NTUDATA\\nturgbd_skeletons\\mat\\"
# 使用其余点减去中心点的距离
def getXViewTrainingAddr():
    train_set=[]
    for i in range(1,61):
        print(i)
        pattern = "*C0%02d*A0%02d.skeleton.mat" % (1, i)
        action_addr = sorted(glob.glob(Dir+pattern))  # all data
        select = np.random.permutation(len(action_addr))[:n_support+n_query]
        for s in select:
            train_set.append(action_addr[s])
    return train_set

def getXViewTestingAddr():
    test_set=[]
    for i in range(1,61):
        pattern = "*C0%02d*A0%02d.skeleton.mat" % (2, i)
        action_addr_1 = sorted(glob.glob(Dir+pattern))
        test_set.append(action_addr_1)
        pattern = "*C0%02d*A0%02d.skeleton.mat" % (3, i)
        action_addr_2 =sorted(glob.glob(Dir+pattern))# all data
        test_set.append(action_addr_2)
    return test_set

def resize(diff_feature):
    sample=np.zeros([im_height,im_width,3])
    sample[:, :, 0] = transform.resize(diff_feature[:, :, 0], (im_height,im_width), mode='reflect', anti_aliasing=True)
    sample[:, :, 1] = transform.resize(diff_feature[:, :, 1], (im_height,im_width ), mode='reflect', anti_aliasing=True)
    sample[:, :, 2] = transform.resize(diff_feature[:, :, 2], (im_height,im_width ), mode='reflect', anti_aliasing=True)
    return sample
# 使用其余点减去中心点的距离
def get_diff_feature(feature,ref_point_index=1):

    for i in range(feature.shape[1]):
        feature[:,i,:]=feature[:,i,:]-np.repeat(np.expand_dims(feature[ref_point_index, i, :], axis=0),feature.shape[0],axis=0)
    im=np.delete(feature,ref_point_index,axis=0)
    factor=max_diff_channal(im)
    for i in range(im.shape[2]):
        im[:,:,i]=Normalize(im[:,:,i],factor)
    sample=resize(im)
    return sample
def ouput_3_gray_imge(diff_feature,path):
    prename = path.split('\\')[-1]
    print(prename)
    x_im=diff_feature[:, :, 0]*255
    im = Image.fromarray(x_im.astype(np.uint8))
    im.save((".\\data\\Skeleton4\\data\\x_%s.bmp") % (prename))

    y_im=diff_feature[:, :, 1]*255
    im = Image.fromarray(y_im.astype(np.uint8))
    im.save((".\\data\\Skeleton4\\y_%s.bmp") % (prename))

    z_im=diff_feature[:, :, 2]*255
    im = Image.fromarray(z_im.astype(np.uint8))
    im.save((".\\data\\Skeleton4\\z_%s.bmp") % (prename))

def prepar_train_data(data_addr, n_classes):
    train_data_set = np.zeros([n_classes, n_query + n_support, im_height, im_width, 3], dtype=np.float32)
    sample_index=np.zeros(60,dtype=int)
    for addr in data_addr:
        action_index, sample = toImageSample(addr)
        train_data_set[action_index,sample_index[action_index]]=sample
        sample_index[action_index]=sample_index[action_index]+1
    return train_data_set


def toImageSample(addr):
    skelet = load_data(addr)  # skelet是numpy的ndarray类型
    token = addr.split('\\')[-1].split('.')[0]
    action_index = int(token[17:]) - 1
    bodyA = skelet[0, :, :, :]
    bodyB = skelet[1, :, :, :]
    sample = None
    # 拼接两个图像
    if bodyA.max() > 1e-6 and bodyB.max() > 1e-6:
        imB = get_diff_feature(bodyB, 1)
        imA = get_diff_feature(bodyA, 1)
        merge = np.vstack((imA, imB))
        sample = resize(merge)
    elif bodyA.max() > 1e-6:
        sample = get_diff_feature(bodyA, 1)
    elif bodyB.max() > 1e-6:
        sample = get_diff_feature(bodyB, 1)
    else:
        print("error")
    # 根据具体的类存入到相应的位置中
    ouput_3_gray_imge(sample, token)
    return action_index, sample


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
        block_2_out = tf.layers.conv2d(block_2_in, h_dim*2, kernel_size=[2, 3], dilation_rate=[2, 2],padding='SAME')
        # block_2_out = tf.contrib.layers.batch_norm(block_2_out, updates_collections=None, decay=0.99, scale=True,center=True)
        block_2_out = tf.nn.relu(block_2_out)
        #---------#

        #---------#
        block_3_in = tf.concat([block_2_out, block_1_out,block_1_in], axis=3)
        block_3_out = tf.layers.conv2d(block_3_in, h_dim*3, kernel_size=[2, 3], dilation_rate=[2, 2],padding='SAME')
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
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [1,2],strides=[1, 2])
        net = tf.layers.conv2d(net, h_dim*4, kernel_size=5,padding='SAME')  # 64 filters, each filter will generate a feature map.
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [2, 3], strides=[2, 3])
        net = tf.layers.conv2d(net, h_dim*2, kernel_size=5,padding='SAME')  # 64 filters, each filter will generate a feature map.
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, [2, 3], strides=[2, 3])
        #dense
        net = tf.layers.flatten(net)#tf.contrib.layers.flatten(P)这个函数就是把P保留第一个维度，把第一个维度包含的每一子张量展开成一个行向量，返回张量是一个二维的

        return net





def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def train_test():
    # train_data_addr=getXViewTrainingAddr()
    # for i in  train_data_addr:
    #     tmp=i.replace("mat","mat_train")
    #     mymovefile(i,tmp)
    train_data_addr=glob.glob('D:\\NTUDATA\\nturgbd_skeletons\\mat_train\\*.mat_train')
    train_dataset=prepar_train_data(train_data_addr, n_classes)
    print(train_dataset.shape)#(60,10, 25, 90, 3)


    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels],name='x')
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels],name='q')
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    #训练的时候具有support sample的参数
    num_classes, num_support = x_shape[0], x_shape[1]# num_class num_support_sample
    num_queriy_class,num_queries = q_shape[0],q_shape[1]#num_query_sample
    #y为label数据由外部导入
    y = tf.placeholder(tf.int64, [None, None],name='y')
    y_one_hot = tf.one_hot(y, depth=num_classes)# dimesion of each one_hot vector
    #emb_x是样本通过encoder之后的结果
    emb_x = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim,reuse=False)
    emb_dim = tf.shape(emb_x)[-1] # the last dimesion

    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)#计算每一类的均值，每一个类的样本都通过CNN映射到高维度空间
    emb_q = encoder(tf.reshape(q, [num_queriy_class * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)

    dists = euclidean_distance(emb_q, emb_x)

    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])#-1表示自动计算剩余维度，paper中公式2 log_softmax 默认 axis=-1
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]),name='loss')#reshpae(a,[-1])会展开所有维度, ce_loss=cross entropy
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)),name='acc')

    tf.add_to_collection('acc', acc)
    tf.add_to_collection('loss', ce_loss)
    train_op = tf.train.AdamOptimizer().minimize(ce_loss)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
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
    total_ls=0
    corecct_count=0
    total_count=0
    test_set = getXViewTestingAddr()
    while True:
        epi_classes = (np.arange(n_classes))[:n_test_way]
        support = np.zeros([n_test_way, n_test_support, im_height, im_width, channels], dtype=np.float32)
        #query = np.zeros([n_test_way, n_test_query, im_height, im_width,channels], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes):
            selected_support = np.random.permutation(n_query+n_support)[:n_test_support]#从训练集合取support样本
            #selected_query = np.random.permutation(n_test_query)#22个样本
            support[i] = train_dataset[epi_cls, selected_support]#从训练集合取support样本
            #query[i] = test_dataset[epi_cls, selected_query]
        # support = np.expand_dims(support, axis=-1)
        # query = np.expand_dims(query, axis=-1)
        '''
        1 1 1 
        2 2 2
        3 3 3
        4 4 4
        5 5 5
        '''
        #id, count, query = nextTestSample()
        for addr in test_set:
            action_index, sample=toImageSample(addr)
            action=action_index+1
            labels = np.tile(np.array([action])[:, np.newaxis], (1, 1)).astype(np.uint8)
            ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y:labels})
            corecct_count+=ac*1
            total_count+=1
            total_ls+= (ls*1)
            print(corecct_count)
        # avg_acc += ac
        # avg_ls+=ls
        # if (epi+1) % 50 == 0:
        #     print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f} '.format(epi+1, n_test_episodes, ls, ac))
    avg_acc = corecct_count/total_count
    avg_ls=total_ls/total_count
    print('Average Test Accuracy: {:.5f} Average loss : {:.5f}'.format(avg_acc,avg_ls))

    '''
    #， 训练样本的修改：现在每次的都是从32个样本中随机抽取5个作为支持向量，5个作为query向量。能否改成只有在10个样本中进行随机抽取， 完成
    #， 测试的修改：分27类 现在每次测试都是从32个样本里面随机抽5个当作支持向量，检验剩余27样本的数据，
    图片的生成：现在每次都是从sequnce中生成一张图片，能否生成多张图片？
    '''
# def load_test():
#
#     data_addr = sorted(glob.glob('.\\data\\Skeleton\\data\\*.mat'))  # all data
#     test_dataset, train_dataset = prepar_data(data_addr, n_classes)
#     print(test_dataset.shape)
#     sess = tf.Session()
#     saver = tf.train.import_meta_graph('%s.meta'%ckpt_path)
#     saver.restore(sess,ckpt_path)
#     graph = tf.get_default_graph()
#     x=graph.get_operation_by_name('x').outputs[0]
#     y=graph.get_operation_by_name('y').outputs[0]
#     q =graph.get_operation_by_name('q').outputs[0]
#     ce_loss=tf.get_collection('loss')[0]
#     acc=tf.get_collection('acc')[0]
#     avg_acc = 0.
#     avg_ls=0.
#     for epi in range(n_test_episodes):
#         epi_classes = np.random.permutation(n_classes)[:n_test_way]
#         support = np.zeros([n_test_way, n_test_support, im_height, im_width, channels], dtype=np.float32)
#         query = np.zeros([n_test_way, n_test_query, im_height, im_width,channels], dtype=np.float32)
#         for i, epi_cls in enumerate(epi_classes):
#
#             selected_support = np.random.permutation(n_query+n_support)[:n_test_support]#从训练集合取support样本
#             selected_query = np.random.permutation(n_test_query)#22个样本
#             support[i] = train_dataset[epi_cls, selected_support]#从训练集合取support样本
#             query[i] = test_dataset[epi_cls, selected_query]
#         # support = np.expand_dims(support, axis=-1)
#         # query = np.expand_dims(query, axis=-1)
#         labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
#         ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y:labels})
#
#         avg_acc += ac
#         avg_ls+=ls
#         if (epi+1) % 50 == 0:
#             print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f} '.format(epi+1, n_test_episodes, ls, ac))
#     avg_acc /= n_test_episodes
#     avg_ls/=n_test_episodes
#     print('Average Test Accuracy: {:.5f} Average loss : {:.5f}'.format(avg_acc,avg_ls))
train_test()