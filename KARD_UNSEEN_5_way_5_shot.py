import KARD_UNSeen
import numpy as np
import os
KARD_UNSeen.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD_UNSeen.n_episodes=200
KARD_UNSeen.n_train_classes=8
KARD_UNSeen.n_test_classes=10

KARD_UNSeen.n_way=5
KARD_UNSeen.n_support=5
KARD_UNSeen.n_query=5

KARD_UNSeen.n_test_way=5
KARD_UNSeen.n_test_support=5
KARD_UNSeen.n_test_query=KARD_UNSeen.n_sample_per_class-KARD_UNSeen.n_test_support

KARD_UNSeen.AS_Train=np.array([0,1,2,3,4,5,6,7])
KARD_UNSeen.AS_Test=np.array([8,9,10,11,12,13,14,15,16,17])
KARD_UNSeen.train_test()