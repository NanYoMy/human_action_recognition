#-*- encoding=utf-8 -*-
import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow as tf
import glob
from myanimation import AnimatedScatter
from mpl_toolkits.mplot3d import Axes3D
from skimage import transform,io
_resize_width=40
_resize_height=60
# 可以通过mat数据让matlab代码与tensorflow代码链接起来
def load_data(path):
    data = sio.loadmat(path)
    skelet=data['d_skel']
    print(skelet.shape)#(20,3,48) 20个点，每个点有xyz, 一共48帧，这里需要处理不帧数的样本
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
def get_diff_feature(skelet):
    feature=skelet.swapaxes(1,2)
    ref_point_index=3;
    for i in range(feature.shape[1]):
        feature[:,i,:]=feature[:,i,:]-np.repeat(np.expand_dims(feature[ref_point_index, i, :], axis=0),feature.shape[0],axis=0)
    im=np.delete(feature,2,axis=0)
    factor=max_diff_channal(im)
    for i in range(im.shape[2]):
        im[:,:,i]=Normalize(im[:,:,i],factor)
    sample=resize(im)
    return sample

# 单独的使用skelet作为feature
def get_diff_feature3(skelet):
    feature = skelet.swapaxes(1, 2)
    factor=max_diff_channal(feature)
    for i in range(skelet.shape[1]):
        feature[:,:,i]=Normalize(feature[:,:,i],factor)
    return feature.astype(np.uint8)

# 使用两两帧差的数据并进行归一化处理
def get_diff_feature2(skelet):
    im = skelet.swapaxes(1, 2)
    feature=np.zeros([im.shape[0],im.shape[1]-1,im.shape[2]])
    for i in range(im.shape[1]-1):
        feature[:,i,:]=im[:,i+1,:]-im[:,i,:]
    factor=max_diff_channal(feature)
    for i in range(feature.shape[2]):
        feature[:,:,i]=Normalize(feature[:,:,i],factor)
    return feature

# 使用两两帧差的binary feature  失败
def get_diff_feature_sigmod(skelet):
    im = skelet.swapaxes(1, 2)
    feature=np.zeros([im.shape[0],im.shape[1]-1,im.shape[2]])
    for i in range(im.shape[1]-1):
        feature[:,i,:]=im[:,i+1,:]-im[:,i,:]

    feature=np.where(feature>0,255,0)
    feature = 1.0 / (1.0 + 1.0 / np.exp(-feature))*255

    return feature

def ouput_3_gray_imge(diff_feature,path):
    prename = path.split('\\')[-1]
    x_im=transform.resize(diff_feature[:, :, 0], (_resize_width, _resize_height), mode='reflect',anti_aliasing=True)
    x_im=x_im*255
    im = Image.fromarray(x_im.astype(np.uint8))
    im.save((".\\data\\Skeleton\\subdata\\x_%s.bmp") % (prename))

    y_im=transform.resize(diff_feature[:, :, 1], (_resize_width, _resize_height), mode='reflect',anti_aliasing=True)
    y_im=y_im*255
    im = Image.fromarray(y_im.astype(np.uint8))
    im.save((".\\data\\Skeleton\\subdata\\y_%s.bmp") % (prename))

    z_im=transform.resize(diff_feature[:, :, 2], (_resize_width, _resize_height), mode='reflect',anti_aliasing=True)
    z_im=z_im*255
    im = Image.fromarray(z_im.astype(np.uint8))
    im.save((".\\data\\Skeleton\\subdata\\z_%s.bmp") % (prename))
    print(".\\data\\Skeleton\\subdata\\%s.bmp" % (prename))
def ouput_RGB_imge(diff_feature,path):
    rgb_image = transform.resize(diff_feature, (_resize_width, _resize_height, 3))
    rgb_image=rgb_image*255
    im = Image.fromarray(rgb_image.astype(np.uint8))
    prename = path.split('\\')[-1]
    im.save((".\\data\\Skeleton\\subdata\\%s.bmp") % (prename))
    print(".\\data\\Skeleton\\subdata\\%s.bmp" % (prename))

def resize(diff_feature):
    sample=np.zeros([_resize_width,_resize_height,3])
    sample[:,:,0] = transform.resize(diff_feature[:, :, 0], (_resize_width, _resize_height), mode='reflect', anti_aliasing=True)
    sample[:, :, 1] = transform.resize(diff_feature[:, :, 1], (_resize_width, _resize_height), mode='reflect', anti_aliasing=True)
    sample[:, :, 2] = transform.resize(diff_feature[:, :, 2], (_resize_width, _resize_height), mode='reflect', anti_aliasing=True)
    return sample


n_sample_per_class=32
n_class=10
def getSample():
    data_addr = sorted(glob.glob('.\\data\\Skeleton\\subdata\\*.mat'))
    train_data_set=np.zeros([n_class,n_sample_per_class,_resize_width, _resize_height,3])
    for j in range(len(data_addr)):
        print(data_addr[j])
        skelet = load_data(data_addr[j])# skelet是numpy的ndarray类型
        print(data_addr[j])
        token = data_addr[j].split('\\')[-1].split('_')
        i=int(token[0][1:])-1
        j=(int(token[1][1:])-1)*4+int(token[2][1:])-1
        sample=get_diff_feature(skelet)
        print("%d,%d"%(i,j))
        train_data_set[i,j]=sample
        # print(diff_feature)
        ouput_3_gray_imge(sample,data_addr[j])
        ouput_RGB_imge(sample, data_addr[j])
        # a = AnimatedScatter(skelet,skelet.shape[2])
        # a.show()
    return train_data_set


getSample()






