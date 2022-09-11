import json
import os, time
import glob
from tqdm import tqdm
import numpy as np
from metrics import evaluation_metric
from trainer import Trainer
from config import TAGS
import pandas as pd

root = 'final_models'
names = ["test_all_4_100k", "test_all_2_100k", "test_all_5_100k"]
models = [557, 705, 358]

MIN_MODEL = 652
N_SAMPLES = 10000
BATCH_SIZE = 50
IMAGE_SIZES = [128, 128, 128]
STEP = 1
N_STATS = 10
TRUNC = 1
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"
REAL_EMBEDS = {"FID": FOLDER + "\\real_embeddings_inception.pt", "PR": FOLDER + "\\real_embeddings_vgg_pr_re.pt"}

f1_score = lambda x, y: (2 * x * y) / (x + y) if (x+y) != 0 else 0.0

for i, [name, m] in enumerate(zip(names, models)):

    with open(os.path.join(root, name, 'config.json'), 'r') as file:
        config = json.load(file)
    model = Trainer(**config)
    model.load(m, root=root)

    fid, precision, recall, f1_s = [], [], [], []
    for j in tqdm(range(N_STATS)):
        fid.append(evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZES[i], num_samples=N_SAMPLES,
                                     use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                     verbose=False))
        p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZES[i], num_samples=N_SAMPLES,
                                 use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                 verbose=False)
        precision.append(p)
        recall.append(r)
        f1_s.append(f1_score(p, r))

    fid_mean, fid_std = np.mean(fid), np.std(fid)
    p_mean, p_std = np.mean(precision), np.std(precision)
    r_mean, r_std = np.mean(recall), np.std(recall)
    f1_mean, f1_std = np.mean(f1_s), np.std(f1_s)

    with open(f'mean_std_metrics.csv', 'a+') as file:
        if i == 0:
            file.write("model,fid_mean,fid_std,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std\n")
        file.write(str(m) + "," + str(fid_mean) + "," + str(fid_std) + "," + str(p_mean) + "," + str(p_std) + "," +
                   str(r_mean) + "," + str(r_std) + "," + str(f1_mean) + "," + str(f1_std) + "\n")
    break
print(pd.read_csv("mean_std_metrics.csv"))







# model.load(378, root=root)
# start = time.time()
# # fid = evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
# #                         use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
# #                         verbose=True)
# # print("\nTiempo de ejecución FID: ", time.time() - start)
# p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
#                          use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
#                          verbose=True)
# print("\nTiempo de ejecución PR: ", time.time() - start)
# print(f"FID: {0}\nP: {p},\nR: {r}\nF1: {f1_score(p, r)}")

"""
# Get number of models finished in .pt
num_models = len(glob.glob(os.path.join(root, name, '*.pt')))
# num_models = 454
progress = tqdm(reversed(range(MIN_MODEL, num_models // STEP)))
for m in progress:
    model.load((m * STEP), root=root)
    fid = []
    p_row, r_row = [], []
    for i in range(N_STATS):

        fid.append(evaluation_metric("FID", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                     use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                     verbose=False))

        p, r = evaluation_metric("PR", CSV_PATH, FOLDER, model, IMAGE_SIZE, num_samples=N_SAMPLES,
                                 use_labels=False, tags=TAGS, batch_size=BATCH_SIZE, truncation_trick=TRUNC,
                                 verbose=False)
        p_row.append(p)
        r_row.append(r)
        progress.set_description("FID: %.4f, P: %.4f, R: %.4f | Model %3d / %3d" %
                                 (np.nanmean(fid), np.nanmean(p_row), np.nanmean(r_row),
                                  num_models - m // STEP, num_models // STEP - MIN_MODEL))

    fid_mean, p_mean, r_mean = np.nanmean(fid), np.nanmean(p_row), np.nanmean(r_row)
    fid_std, p_std, r_std = np.nanstd(fid), np.nanstd(p_row), np.nanstd(r_row)

    with open(os.path.join(root, name, f'metrics_{name}.csv'), 'a+') as file:

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
"""