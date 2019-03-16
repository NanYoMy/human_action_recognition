#-*- encoding=utf8 -*-
#https://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

def create_sprite(images):
    if isinstance(images,list):
        images=np.array(images)
    img_h=images.shape[1]
    img_w=images.shape[2]
    n_plots=int(np.ceil(np.sqrt(images.shape[0])))#100=10*10
    spriteimage=np.ones((n_plots*img_h,n_plots*img_w))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter=i*n_plots+j
            if this_filter<images.shape[0]:
                this_img=images[this_filter]
                spriteimage[i*img_h:(i+1)*img_h,j*img_w:(j+1)*img_w]=this_img
    return spriteimage
def verctor_to_matrix(vector):
    return np.reshape(vector,(-1,28,28))
def invert_grayscale(img):
    return 1-img

LOG_DIR="log"
NAME_TO_VISUALISE_VARIABLE="mnistembbeding"
#What to visualise
TO_EMBBD_COUNT=500
path_for_mnist_sprites=os.path.join(LOG_DIR,"mnistdigit.png")
path_for_mnist_metadata=os.path.join(LOG_DIR,"metadata.tsv")

mnist=input_data.read_data_sets("data/minist",one_hot=True)
batch_xs,batch_ys=mnist.train.next_batch(TO_EMBBD_COUNT)

#Creating the embeddings
embbeding_var=tf.Variable(batch_xs,name=NAME_TO_VISUALISE_VARIABLE)
summary_writer=tf.summary.FileWriter(LOG_DIR)

#create the embbeding projectorc
config = projector.ProjectorConfig()
embbeding=config.embeddings.add()
embbeding.tensor_name=embbeding_var.name

embbeding.metadata_path=path_for_mnist_metadata
embbeding.sprite.image_path=path_for_mnist_sprites
embbeding.sprite.single_image_dim.extend([28,28])

projector.visualize_embeddings(summary_writer,config)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
saver.save(sess,os.path.join(LOG_DIR,"model.ckpt"),1)

to_visualise = batch_xs
to_visualise = verctor_to_matrix(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite(to_visualise)

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')











