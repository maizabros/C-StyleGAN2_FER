from pathlib import Path
from torch import cuda

GPU_BATCH_SIZE = 8
GRADIENT_BATCH_SIZE = 32
PATH_LENGTH_REGULIZER_FREQUENCY = 2
GRADIENT_ACCUMULATE_EVERY = 8 # unused if using run.py


CONDITION_ON_MAPPER = False
CHANNELS = 3
NETWORK_CAPACITY = 16
STYLE_DEPTH = 8
IMAGE_SIZE = 128
LATENT_DIM = 512
USE_BIASES = False
USE_COROPPED = 1

EXTS = ['jpg', 'png']
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet\\affectnet_complete.csv"
TAGS = ["label", "age", "gender", "race", "race4"]
IGNORE_TAGS = ["partition", "subject", "sequence", "cropped_img", "age_conf_fair", "age_scores_fair",
               "gender_conf_fair", "gender_scores_fair", "race_conf_fair", "race_scores_fair", "race_conf_fair_4",
               "race_scores_fair_4"]
APL_PATH = "all_possible_labels.pt"
CATEGORIES = len(TAGS)
HOMOGENEOUS_LATENT_SPACE = True
USE_DIVERSITY_LOSS = True
MIXED_PROBABILITY = 0.9

MOVING_AVERAGE_START = 4000
MOVING_AVERAGE_PERIOD = 200


EVALUATE_EVERY = 100
STEP_FACTOR = 2000
SAVE_EVERY = 250
NUM_TRAIN_STEPS = EVALUATE_EVERY * STEP_FACTOR

NAME = "test_all_2_100k_bis"
CURRENT_DIR = Path('.')
LOG_FILENAME = CURRENT_DIR / f'logs_{NAME}.csv'
MODELS_DIR = CURRENT_DIR / 'models'
RESULTS_DIR = CURRENT_DIR / 'results'

NEW = False
LOAD_FROM = -1

DEVICE = "cuda" if cuda.is_available() else "cpu"
GPU = '0'
EPSILON = 1e-8
LEARNING_RATE = 2e-4
LABEL_EPSILON = 0
