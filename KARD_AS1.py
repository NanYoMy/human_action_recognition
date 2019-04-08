import numpy as np
import  KARD
import os
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.n_episodes=80
KARD.is_sub_set=True
KARD.AS=np.array([0,2,11,14,17,8,13,5])
KARD.load_test()
