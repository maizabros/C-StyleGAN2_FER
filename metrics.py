import numpy as np
import torchvision.utils
from scipy import linalg
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from InceptionV3 import InceptionV3
from dataset import cycle
from utils import GenDataset, RealDataset
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

AVAILABLE_METRICS = ["FID", "PR"]


def compute_embeddings(dataloader, count, batch_size, embedding_model, num_features, metric, verbose=False,
                       reals_preload=None):
    if reals_preload is not None:
        print("     Loading precomputed embeddings...", end="") if verbose else None
        image_embeddings = torch.load(reals_preload[metric])
        return image_embeddings[torch.randperm(image_embeddings.size(0))[:count]]

    image_embeddings = None
    if metric == "FID":
        image_embeddings = np.zeros((count * batch_size, num_features))
    elif metric == "PR":
        image_embeddings = torch.zeros((count * batch_size, num_features))
    for i in tqdm(range(count)) if verbose else range(count):
        images = next(iter(dataloader)).cuda()
        embeddings = embedding_model(images)
        embeddings = embeddings[0].detach().cpu().numpy() if metric == "FID" else embeddings.detach().cpu()
        image_embeddings[i * batch_size:(i + 1) * batch_size] = embeddings.reshape((batch_size, num_features))
    return image_embeddings


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def manifold_estimate(A_features, B_features, k=3):

    KNN = NearestNeighbors(metric="precomputed", n_neighbors=k)
    KNN_list_in_A = KNN.fit(pairwise_distances(A_features, n_jobs=-1)).kneighbors()[0][:, k-1]
    B_A_distances = pairwise_distances(B_features, A_features, n_jobs=-1)
    nearest_A = np.argmin(B_A_distances, axis=1)
    n = np.sum(B_A_distances[np.arange(len(B_A_distances)), nearest_A] <= KNN_list_in_A[nearest_A])

    return n / len(B_features)


def calculate_pr(real_embeddings, generated_embeddings):
    precision = manifold_estimate(real_embeddings, generated_embeddings)
    recall = manifold_estimate(generated_embeddings, real_embeddings)

    return precision, recall


def get_data_loaders(real_dataset, root, generator_model, image_size, num_samples=10000, batch_size=32,
                     truncation_trick=1.42, use_labels=False, tags=None, use_mapper=True, verbose=False):
    print("Generating datasets...", end="") if verbose else None
    dataset_reals = RealDataset(real_dataset, root, image_size, num_samples=num_samples, tags=tags,
                                use_labels=use_labels)
    dataset_generator = GenDataset(generator_model, num_samples=num_samples, use_labels=use_labels,
                                   labels=dataset_reals.__get_labels__(), truncation_trick=truncation_trick,
                                   use_mapper=use_mapper, image_size=image_size)

    realsloader = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=batch_size,
                                   drop_last=True, shuffle=False, pin_memory=False))
    genloader = cycle(DataLoader(dataset_generator, num_workers=0, batch_size=batch_size,
                                 drop_last=True, shuffle=False, pin_memory=False))
    print("  Done") if verbose else None
    return realsloader, genloader


def evaluation_metric(metric, real_dataset, root, generator_model, image_size, num_samples=10000, batch_size=32,
                      truncation_trick=1.42, use_labels=False, tags=None, reals_preload=None, use_mapper=True,
                      verbose=False):
    embedding_model, calculate_metric, num_features = None, None, None

    if metric not in AVAILABLE_METRICS:
        raise ValueError("Metric {} is not supported".format(metric))
    elif metric == "FID":
        embedding_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda()
        embedding_model.eval()
        calculate_metric = calculate_fid
        num_features = 2048
    elif metric == "PR":
        embedding_model = models.vgg16(pretrained=True)
        embedding_model.classifier = nn.Sequential(*[embedding_model.classifier[i] for i in range(5)])
        embedding_model = embedding_model.cuda().eval()
        calculate_metric = calculate_pr
        num_features = 4096

    count = num_samples // batch_size
    realsloader, genloader = get_data_loaders(real_dataset, root, generator_model, image_size, num_samples=num_samples,
                                              batch_size=batch_size, truncation_trick=truncation_trick,
                                              use_labels=use_labels, tags=tags, use_mapper=use_mapper, verbose=verbose)

    print("Calculating real embeddings...", end="") if verbose else None
    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(realsloader, count, batch_size, embedding_model, num_features, metric,
                                               reals_preload=reals_preload)
    print("  Done") if verbose else None
    print("Calculating generated embeddings...", end="") if verbose else None
    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(genloader, count, batch_size, embedding_model, num_features, metric)
    print("  Done") if verbose else None
    print("Calculating {}...".format(metric), end="") if verbose else None
    return calculate_metric(real_image_embeddings, generated_image_embeddings)
