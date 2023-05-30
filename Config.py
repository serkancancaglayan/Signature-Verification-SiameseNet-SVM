import torch



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#DATA

TRAIN_FOLDER = "./data/train"
TRAIN_ANNOTATION = "./data/train_data.csv"

TEST_FOLDER = "./data/test"
TEST_ANNOTATION = "./data/test_data.csv"

PIN_MEMORY = True
NUM_WORKERS = 4
IMG_SIZE = 224

#MODEL

#BlockType, Inchannels, Outchannels, kernel_size, stride, padding
#BC -> Basic Conv
#DW -> Depth-wise Conv
ARCHITECTURE_CFG = [
        ["BC", 3, 32, 3, 2, 0], 
        ["DW", 32, 64, 3, 1, 0], 
        ["DW", 64, 128, 3, 2, 0],
        ["DW", 128, 128, 3, 1, 0],
        ["DW", 128, 256, 3, 2, 0],
        ["DW", 256, 256, 3, 1, 0],
        ["DW", 256, 512, 3, 2, 0],
        ["DW", 512, 512, 3, 1, 0],
        ["DW", 512, 512, 3, 1, 0],
        ["DW", 512, 512, 3, 1, 0],
        ["DW", 512, 512, 3, 1, 0],
        ["DW", 512, 512, 3, 1, 0],
        ["DW", 512, 1024, 3, 2, 0],
        ["DW", 1024, 1024, 3, 1, 0],
    ]



#OPTIMIZER
LEARNING_RATE = 1e-4
EPS = 1e-8
WEIGHT_DECAY = 5E-4
MOMENTUM = 0.9

#LR SCHEDULER
STEP_SIZE = 5
GAMMA = 0.1

#LOSS
ALPHA = 1.0
BETA = 1.0
MARGIN = 1.0

#TRAINING
BATCH_SIZE = 32
NUM_EPOCHS = 100
SAVE_CHECKPOINT_INTERVAL = 10
CHECKPOINT_PATH = "./checkpoints/"