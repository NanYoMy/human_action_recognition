import UTD_MHAD
import os
UTD_MHAD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
UTD_MHAD.n_episodes=300
UTD_MHAD.train_test()