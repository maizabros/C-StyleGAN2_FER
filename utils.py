import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from config import IMG_LIMIT


class CSVImageDataset(Dataset):

    def __init__(self, csv_file_path, root, image_size, tags, ignore_tags=None, one_hot=True, dropna=True):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        if dropna:
            self.annotations = self.annotations.dropna()
        if IMG_LIMIT > 0:
            self.annotations = self.annotations.sample(n=IMG_LIMIT)
        if ignore_tags:
            assert sum([ignore_tags[i] not in self.annotations.columns for i in range(len(ignore_tags))]) == 0, \
                "Error checking tags in Dataframe Keys. Check if all of the tags exists in your CSV file"
            self.annotations = self.annotations.drop(ignore_tags, axis=1)
        # Encode if wanted
        if one_hot:
            labels_encoded = torch.tensor(np.array([
                LabelEncoder().fit_transform(self.annotations.loc[:, i])
                for i in tags  # except ID and file name
            ]), dtype=torch.int64).T  # needed int64 for F.one_hot
            # Encode to one_hot
            self.labels = torch.cat([
                F.one_hot(labels_encoded[:, i])
                for i in range(labels_encoded.shape[1])
            ], dim=-1)
        else:
            self.labels = torch.tensor(self.annotations.iloc[:, 2:])
        self.root = root
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.labels.shape[0]

    def __label_len__(self):
        return self.labels.shape[1]

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotations.iloc[index, 1])
        image = Image.open(image_path)

        labels = self.labels[index].cuda().float()
        if self.transform:
            image = self.transform(image)

        return image, labels


class GenDataset(Dataset):
    def __init__(self, model, num_samples=10000, truncation_trick=1.42, use_mapper=True):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.truncation_trick = truncation_trick
        self.use_mapper = use_mapper


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self.model.set_evaluation_parameters(reset=True, total=1)
        _, average_generated_image = self.model.evaluate(use_mapper=self.use_mapper,
                                                         truncation_trick=self.truncation_trick)
        return average_generated_image[0]


class RealDataset(Dataset):
    def __init__(self, csv_file_path, root, image_size, num_samples=10000, dropna=True):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        if dropna:
            self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.sample(n=num_samples)
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotations.iloc[index, 1])
        image = Image.open(image_path)
        return self.transform(image)
