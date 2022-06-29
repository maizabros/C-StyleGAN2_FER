import json
import os
from tqdm import tqdm
import numpy as np
from metrics import evaluation_metric
from trainer import Trainer

root = 'models'
name = "test_all_2_100k"

N_SAMPLES = 10000
BATCH_SIZE = 50
IMAGE_SIZE = 128
STEP = 1
N_STATS = 1
TRUNC = 1
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"

f1_score = lambda x, y: (2 * x * y) / (x + y)

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)

# model.load(-1, root=root)
# fid_row = []
# for i in range(N_STATS):
#     fid_row.append(evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
#                                      batch_size=BATCH_SIZE, truncation_trick=TRUNC, verbose=True))
# print(np.array(fid_row))

for m in tqdm(range((len(os.listdir(os.path.join(root, name)))-1) // STEP)):

    model.load((m * STEP), root=root)
    for i in range(N_STATS):
        fid = evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                batch_size=BATCH_SIZE,
                                truncation_trick=TRUNC, verbose=False)
        p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES // 50,
                                 batch_size=BATCH_SIZE,
                                 truncation_trick=TRUNC, verbose=False)
        with open(os.path.join(root, name, 'fid_stats.csv'), 'a+') as file:
            file.write(str(m * STEP) + "," + str(np.array(fid).mean()) + "," + str(np.array(p).mean()) + "," +
                       str(np.array(r).mean()) + "," + str(f1_score(p, r)) + "\n")
