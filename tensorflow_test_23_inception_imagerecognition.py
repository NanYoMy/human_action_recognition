#-*- encoding=utf8 -*-
import tensorflow as tf
import os
import numpy
import matplotlib.pyplot as plt
import numpy as np
from PIL import  Image
with tf.gfile.FastGFile('pre_train_model/inception/classify_image_graph_def.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')

with tf.Session() as sess:
    softmax_tensor=sess.graph.get_tensor_by_name("softmax:0")
    img_data=tf.gfile.FastGFile('pre_train_model/inception/cropped_panda.jpg','rb').read()
    prediction=sess.run(softmax_tensor,{'DecodeJpeg/contents:0':img_data})
    prediction=np.squeeze(prediction)
    top_k=prediction.argsort()[-5:][::-1]
    print(top_k)

    img=Image.open('pre_train_model/inception/cropped_panda.jpg')
    plt.imshow(img)
    plt.axis('off')
    plt.show()