import os

# ----------------------------------------------------
# DATA I/O

TRAIN_DIR = ''
LABEL_DIR = TRAIN_DIR
CHECKPOINT_DIR = ''
TEST_DIR = ''
OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, 'validate_submit/')
COLOR_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, 'validate_submit_color/')
PRETRAINED_MODEL = None
SINGLEVIEW_TEST_MODEL = None
SEMANTIC_TEST_MODEL = None

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.isdir(COLOR_OUTPUT_DIR):
    os.makedirs(COLOR_OUTPUT_DIR)

CONTINUE_TRAINING = False
CONTINUE_SEMANTIC_MODEL_FILE = SEMANTIC_TEST_MODEL
CONTINUE_SINGLEVIEW_MODEL_FILE = SINGLEVIEW_TEST_MODEL

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model_{epoch:02d}.ckpt')

CLASS_FILE_STR = '_CLS'
DEPTH_FILE_STR = '_AGL'
IMG_FILE_STR = '_RGB'
IMG_FILE_EXT = 'tif'
LABEL_FILE_EXT = IMG_FILE_EXT
COLOR_PRED_EXT = 'png'

CLSPRED_FILE_STR = '_CLS'
AGLPRED_FILE_STR = '_AGL'

# for MSI image training
MEAN_VALS_FILE = 'msi_mean.json'
MAX_VAL = 65536  # MSI images are int16, so dividing by this instead of 255

# Option
PYLIBS_DIR = ''

# ----------------------------------------------------
# MODEL TRAINING/TESTING

TEST_GPUS = '0'
LEARNING_RATE = 0.0001
NUM_CHANNELS = 3
BATCH_SZ = 2
IMG_SZ = (1024, 1024)  # this code assumes all images in the training set have the same numbers of rows and columns
IGNORE_VALUE = -10000  # nan is set to this for ignore purposes later
NUM_CATEGORIES = 5  # for semantic segmentation
SUMMARY_SAVE_FREQ = 100 # how many iterations between summary file saves
MODEL_SAVE_PERIOD = 10  # how many epochs between model checkpoint saves
MODEL_MAX_KEEP = 6 # how many models keep when training
NUM_EPOCHS = 200  # total number of epochs to train with
BINARY_CONF_TH = 0.4

BLOCK_IMAGES = False
BLOCK_SZ = (1024, 1024)
BLOCK_MIN_OVERLAP = 20

OPTIMIZER = 'Adam'

# loss function option for semantic segmentation task:
# select from:
#     - categorical_crossentropy
#     - ...
SEMANTIC_LOSS = 'categorical_crossentropy'

# loss function option for height estimation task:
# select from:
#     - mse
#     - mae
#     - scaleInv
#     - mse_mix_grad
#     - scaleInv_mix_grad
#     - ...
SINGLEVIEW_LOSS = 'mse'

BACKBONE = 'resnet_v1_50'
ENCODER_WEIGHTS = 'imagenet'

SINGLEVIEW_CMAP = 'jet'

# ----------------------------------------------------
# FLAGS

SEMANTIC_MODE = 0
SINGLEVIEW_MODE = 1

TRAIN_MODE = 0
TEST_MODE = 1

# ----------------------------------------------------------
# LABEL MANIPULATION

CONVERT_LABELS = True

LAS_LABEL_GROUND = 2
LAS_LABEL_TREES = 5
LAS_LABEL_ROOF = 6
LAS_LABEL_WATER = 9
LAS_LABEL_BRIDGE_ELEVATED_ROAD = 17
LAS_LABEL_VOID = 65

TRAIN_LABEL_GROUND = 0
TRAIN_LABEL_TREES = 1
TRAIN_LABEL_BUILDING = 2
TRAIN_LABEL_WATER = 3
TRAIN_LABEL_BRIDGE_ELEVATED_ROAD = 4
TRAIN_LABEL_VOID = NUM_CATEGORIES

LABEL_MAPPING_LAS2TRAIN = {}
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_GROUND] = TRAIN_LABEL_GROUND
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_TREES] = TRAIN_LABEL_TREES
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_ROOF] = TRAIN_LABEL_BUILDING
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_WATER] = TRAIN_LABEL_WATER
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_BRIDGE_ELEVATED_ROAD] = TRAIN_LABEL_BRIDGE_ELEVATED_ROAD
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_VOID] = TRAIN_LABEL_VOID

LABEL_MAPPING_TRAIN2LAS = {}
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_GROUND] = LAS_LABEL_GROUND
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_TREES] = LAS_LABEL_TREES
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BUILDING] = LAS_LABEL_ROOF
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_WATER] = LAS_LABEL_WATER
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BRIDGE_ELEVATED_ROAD] = LAS_LABEL_BRIDGE_ELEVATED_ROAD

# ----------------------------------------------------------
# SHIELD IMAGES during SINGLEVIEW TRAINING
# mark a few unusual single-view training items

SHIELD_SINGLEVIEW_ITEMS = ['JAX_314_012',
                           'JAX_314_006',
                           'JAX_314_002',
                           'JAX_314_001',
                           'JAX_149_012']
