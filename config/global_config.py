# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : global_config.py
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

__C.TRAIN.EPOCHS = 100010
__C.TRAIN.LEARNING_RATE = 0.002
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.95
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
__C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.IMG_HEIGHT = 240
__C.TRAIN.IMG_WIDTH = 360

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
__C.TEST.BATCH_SIZE = 1
__C.TEST.IMG_HEIGHT = 240
__C.TEST.IMG_WIDTH = 360