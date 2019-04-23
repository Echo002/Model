import os

#
# path and dataset parameter
#

# 存放数据的一系列文件的路径

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')   # cache本身就是隐藏意思

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weight')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

# 实现的20个类别
# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']

CLASSES = ['inflection']  # guaidian

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 512

CELL_SIZE = 16    # cell尺寸

BOXES_PER_CELL = 2   # 每个格 cell负责的boxes类别

ALPHA = 0.1  # 激活函数系数

DISP_CONSOLE = False
# 权重衰减的相关参数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0   # bounding box的5个值


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 500

DECAY_RATE = 0.9  #每DECAY_STEPS就LEARNING_RATE*DECAY_RATE ，更新LEARNING_RATE

STAIRCASE = True #

BATCH_SIZE = 5

MAX_ITER = 15000

SUMMARY_ITER = 10  # 日志记录迭代步数

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.04

IOU_THRESHOLD = 0.5  # IoU 参数
