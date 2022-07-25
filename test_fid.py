import json
import os
import glob
from tqdm import tqdm
import numpy as np
from metrics import evaluation_metric
from trainer import Trainer
from config import TAGS

root = 'models'
name = "test_all_4_100k"

MIN_MODEL = 200
N_SAMPLES = 10000
BATCH_SIZE = 50
IMAGE_SIZE = 128
STEP = 1
N_STATS = 1
TRUNC = 1
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"
REAL_EMBEDS = {"FID": FOLDER + "\\real_embeddings_inception.pt", "PR": FOLDER + "\\real_embeddings_vgg_pr_re.pt"}

f1_score = lambda x, y: (2 * x * y) / (x + y) if (x+y) != 0 else 0.0

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)

# model.load(-1, root=root)
# fid_row = []
# for i in range(N_STATS):
#     fid_row.append(evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
#                                      batch_size=BATCH_SIZE, truncation_trick=TRUNC,
#                                      reals_preload=REAL_EMBEDS, verbose=True))
# print(np.array(fid_row))
# Get number of models finished in .pt
# num_models = len(glob.glob(os.path.join(root, name, '*.pt')))
num_models = 454
progress = tqdm(reversed(range(MIN_MODEL, num_models // STEP)))
for m in progress:
    model.load((m * STEP), root=root)
    fid = []
    p_row, r_row = [], []
    for i in range(N_STATS):

        fid.append(evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                     use_labels=True, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                     verbose=False))

        p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                 use_labels=True, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                 verbose=False)
        p_row.append(p)
        r_row.append(r)
        progress.set_description("FID: %.4f, P: %.4f, R: %.4f | Model %3d / %3d" %
                                 (np.nanmean(fid), np.nanmean(p_row), np.nanmean(r_row),
                                  num_models - m // STEP, num_models // STEP - MIN_MODEL))

    fid_mean, p_mean, r_mean = np.nanmean(fid), np.nanmean(p_row), np.nanmean(r_row)
    fid_std, p_std, r_std = np.nanstd(fid), np.nanstd(p_row), np.nanstd(r_row)

    with open(os.path.join(root, name, f'metrics_{name}_with_labels.csv'), 'a+') as file:

        if N_STATS > 1:
            if m == MIN_MODEL:
                file.write("model,fid_mean,fid_std,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std\n")
            file.write(str(m * STEP) + "," +
                       str(fid_mean) + "," + str(fid_std) + "," +
                       str(p_mean) + "," + str(p_std) + "," +
                       str(r_mean) + "," + str(r_std) + "," +
                       str(f1_score(p_mean, r_mean)) + "," + str(f1_score(p_std, r_std)) + "\n")
        else:
            if m == MIN_MODEL:
                file.write("model,fid,precision,recall,f1\n")
            file.write(str(m * STEP) + "," + str(fid_mean) + "," + str(p_mean) + "," + str(r_mean) + "," +
                       str(f1_score(p_mean, r_mean)) + "\n")
