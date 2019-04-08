import KARD_UNSeen
import os
KARD_UNSeen.ckpt_path='./ckpt/%s'%os.path.basename(__file__)
KARD_UNSeen.n_episodes=100
KARD_UNSeen.load_test()