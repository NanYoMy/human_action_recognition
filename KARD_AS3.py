import numpy as np
import  KARD
import os
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.n_episodes=120
KARD.is_sub_set=True
KARD.AS=np.array([6,15,16,14,10,7,4,0])
KARD.load_test()