# -*- encoding=utf-8 -*-
# matplotlib inline
from __future__ import print_function
import UNSeen


'''
training:利用类编号为1-10的样本,使用5个support样本，利用5个query,对模型进行训练，
inference:利用类编号为11-27的样本,使用5个support样本，利用27个query,对模型进行评估，
'''

UNSeen.n_test_way = 10
UNSeen.train_test()