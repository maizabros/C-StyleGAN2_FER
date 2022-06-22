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

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)

# model.load(-1, root=root)
# fid_row = []
# for i in range(N_STATS):
#     fid_row.append(evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
#                                      batch_size=BATCH_SIZE, truncation_trick=TRUNC, verbose=True))
# print(np.array(fid_row))
models = 235

for m in tqdm(range(models // STEP)):

    model.load(models - (m * STEP), root=root)
    for i in range(N_STATS):
        fid = evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                batch_size=BATCH_SIZE,
                                truncation_trick=TRUNC, verbose=False)
        if (models - m) > 126:
            p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES // 50,
                                     batch_size=BATCH_SIZE,
                                     truncation_trick=TRUNC, verbose=False)
            with open(os.path.join(root, name, 'fid_stats.csv'), 'a+') as file:
                file.write(str(np.array(fid).mean()) + "," + str(np.array(p).mean()) + "," + str(np.array(r).mean()) +
                           "," + str(N_SAMPLES) + "," + str(BATCH_SIZE) + "," + str(IMAGE_SIZE) + "," + str(
                    models - (m * STEP)) + "\n")
        else:
            with open(os.path.join(root, name, 'fid_stats_COP.csv'), 'a+') as file:
                file.write(str(np.array(fid).mean()) + "," + str(m * STEP) + "\n")

        if (models - m) == 36:
            break
    else:
        continue
    break
