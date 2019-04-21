import  SBU
import os
SBU.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
SBU.n_classes=8
SBU.is_sub_set=False
SBU.n_way=SBU.n_classes
SBU.n_episodes=150
SBU.n_query=4
SBU.n_support=4
SBU.n_test_support=4
SBU.train_test()