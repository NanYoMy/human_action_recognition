import numpy as np
import  KARD
import os
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.n_episodes=80
KARD.is_sub_set=True
KARD.AS=np.array([1,9,3,6,12,8,11,16])
KARD.load_test()
