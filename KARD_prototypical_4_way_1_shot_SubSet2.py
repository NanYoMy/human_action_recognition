import  KARD_prototypical as KARD
import os
import numpy as np
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.is_sub_set=True
KARD.AS=np.array([1,9,3,6,12,8,11,16])
KARD.n_way=4
KARD.n_episodes=80
KARD.n_query=1
KARD.n_support=1
KARD.n_test_way=4
KARD.n_test_support=1
KARD.n_test_query=20-KARD.n_test_support
#10个用于训练，20个用于测试
# KARD.load_test()

KARD.train_test()
