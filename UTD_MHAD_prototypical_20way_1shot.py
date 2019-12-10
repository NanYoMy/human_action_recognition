import UTD_MHAD_prototypical as UTD_MHAD
import os
UTD_MHAD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
UTD_MHAD.n_episodes=1000
UTD_MHAD.n_way = 20
UTD_MHAD.n_support=1
UTD_MHAD.n_query=1
UTD_MHAD.n_test_way=20
UTD_MHAD.n_test_support=1
UTD_MHAD.n_test_query=int(UTD_MHAD.n_sample_per_class/2)-UTD_MHAD.n_test_support


UTD_MHAD.train_test()