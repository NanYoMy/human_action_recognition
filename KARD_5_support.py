import  KARD
import os
KARD.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD.n_classes=18
KARD.is_sub_set=False
KARD.n_way=KARD.n_classes
KARD.n_episodes=120

KARD.load_test()

