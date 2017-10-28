from easydict import EasyDict as ed
import os

cfg = ed()

cfg.G1_INPUT_DATA_SHAPE = [256, 256, 21]
cfg.G2_INPUT_DATA_SHAPE = [256, 256, 3]
cfg.BATCH_SIZE = 6
cfg.N = 6