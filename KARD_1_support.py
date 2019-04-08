import  KARD
import os
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.n_classes=18
KARD.is_sub_set=False
KARD.n_way=KARD.n_classes
KARD.n_episodes=150
KARD.n_query=5
KARD.n_support=5
KARD.n_test_support=1

KARD.load_test()

