# -*- encoding=utf8 -*-
import  tensorflow as tf
import matplotlib.pyplot as plt

def getImageTensor(path):
    raw_img=tf.gfile.FastGFile(path,'rb').read()
    return tf.image.decode_bmp(raw_img)

path='img\\P1_1_1_p19.bmp'
img=getImageTensor(path)


with tf.Session() as sess:

    img=sess.run()
    print(img.shape)
    plt.figure()
    plt.imshow(img)
    plt.show()