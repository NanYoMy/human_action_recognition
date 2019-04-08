#-*- encoding=utf-8 -*-
#matplotlib inline
from __future__ import print_function
import UTD_MHAD
'''
training:使用一个support样本，利用7个query,对模型进行训练
inference:使用一个从train样本中得到的support样本，对剩余的24样本进行评估，
#train setting
n_epochs = 20
n_episodes = 90
n_classes=27
n_sample_per_class=32
n_way = n_classes
n_support = 1
n_query = 7
#test setting
n_test_episodes = 1000
n_test_way = n_classes
n_test_support = n_support
n_test_query = n_sample_per_class - n_support - n_query#n_test_shot+n_test_query<=22

'''
import os
UTD_MHAD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
UTD_MHAD.n_support=1
UTD_MHAD.n_query=7
UTD_MHAD.n_test_support=UTD_MHAD.n_support
UTD_MHAD.n_episodes=150
UTD_MHAD.load_test()