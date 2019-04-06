# -*- encoding=utf-8 -*-
# matplotlib inline
from __future__ import print_function
import UTD_MHAD_UNSeen


'''
training:利用类编号为1-10的样本,使用5个support样本，利用5个query,对模型进行训练，
inference:利用类编号为11-27的样本,使用5个support样本，利用27个query,对模型进行评估，
'''
UTD_MHAD_UNSeen.n_episodes=2000
UTD_MHAD_UNSeen.n_test_way = 5
UTD_MHAD_UNSeen.n_way=10
UTD_MHAD_UNSeen.train_test()