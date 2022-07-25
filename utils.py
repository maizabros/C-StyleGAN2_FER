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

    def __init__(self, csv_file_path, root, image_size, tags, ignore_tags=None, one_hot=True, dropna=True,
                 limit_classes=False):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        if dropna:
            self.annotations = self.annotations.dropna()

        if limit_classes:  # Bad implementation!! TODO: fix bad implementation and remove this
            whites = self.annotations[self.annotations.race == "White"].index
            whites2 = self.annotations[self.annotations.race4 == "White"].index
            happy = self.annotations[self.annotations.label == "happy"].index
            neutral = self.annotations[self.annotations.label == "neutral"].index
            age_20_29 = self.annotations[self.annotations.age == "20-29"].index
            men = self.annotations[self.annotations.gender == "Male"].index

            inter_white_happy = np.intersect1d(whites, happy)
            inter_white_neutral = np.intersect1d(whites, neutral)
            inter_white_age_20_29 = np.intersect1d(whites, age_20_29)

            self.annotations = drop_index(self.annotations, inter_white_happy)
            self.annotations = drop_index(self.annotations, inter_white_neutral)
            self.annotations = drop_index(self.annotations, inter_white_age_20_29)
            self.annotations = drop_index(self.annotations, np.random.permutation(men)[:int(len(men)/3.5)])
            self.annotations = drop_index(self.annotations, np.random.permutation(whites)[:int(len(whites)/1.5)])
            self.annotations = drop_index(self.annotations, np.random.permutation(whites2)[:int(len(whites2)/1.5)])
            self.annotations = drop_index(self.annotations, np.random.permutation(happy)[:int(len(happy)/1.25)])
            self.annotations = drop_index(self.annotations, np.random.permutation(neutral)[:int(len(neutral)/1.5)])

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
    def __init__(self, model, num_samples=10000, truncation_trick=1.42, use_labels=False,
                 labels=None, use_mapper=True, image_size=None):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.truncation_trick = truncation_trick
        self.use_mapper = use_mapper
        self.image_size = image_size
        self.use_labels = use_labels
        self.labels = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_labels:
            labels = self.labels[idx].reshape(1, -1).cuda()
            self.model.set_evaluation_parameters(reset=True, labels_to_evaluate=labels, total=1)
        else:
            self.model.set_evaluation_parameters(reset=True, total=1)
        average_generated_image = self.model.evaluate(use_mapper=self.use_mapper, only_ema=True,
                                                      truncation_trick=self.truncation_trick)[0]
        # Resize image if image_size is given to Tensor
        if self.image_size != average_generated_image.shape[-1]:
            average_generated_image = transforms.Resize(self.image_size)(average_generated_image)

        return average_generated_image


class RealDataset(Dataset):
    def __init__(self, csv_file_path, root, image_size, num_samples=-1, use_labels=False, tags=None, dropna=True):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        if dropna:
            self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.sample(n=num_samples) if num_samples > 0 else self.annotations
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        if use_labels:
            if tags is None:
                raise ValueError("No tags given")
            labels_encoded = torch.tensor(np.array([
                LabelEncoder().fit_transform(self.annotations.loc[:, i])
                for i in tags  # except ID and file name
            ]), dtype=torch.int64).T  # needed int64 for F.one_hot
            # Encode to one_hot
            self.labels = torch.cat([
                F.one_hot(labels_encoded[:, i])
                for i in range(labels_encoded.shape[1])
            ], dim=-1)

    def __len__(self):
        return self.annotations.shape[0]

    def __get_labels__(self):
        return self.labels if hasattr(self, "labels") else None

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotations.iloc[index, 1])
        image = Image.open(image_path)
        return self.transform(image)


def drop_index(data, index, inplace=False):
    if np.intersect1d(index, data.index).size == 0:
        return data
    ii = np.intersect1d(index, data.index)
    try:
        data = data.drop(ii, inplace=inplace)
    except:
        print("Some of the indexes are not in the dataframe")
    return data