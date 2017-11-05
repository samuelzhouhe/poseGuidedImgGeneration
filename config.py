from easydict import EasyDict as ed
import os

cfg = ed()

cfg.IMAGE_SHAPE = [256, 256, 3]
cfg.G1_INPUT_DATA_SHAPE = cfg.IMAGE_SHAPE[:2] + [21]
cfg.BATCH_SIZE = 8
cfg.N = 6  # number of resudual blocks
cfg.WEIGHT_DECAY = 0.005
cfg.LAMBDA = 10
cfg.MAXITERATION = 1000000
cfg.LOGDIR = './logs'
cfg.MODE = 'train'
cfg.RESULT_DIR = './result'
cfg.PART = 'g1'