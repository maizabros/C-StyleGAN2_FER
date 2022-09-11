import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms


class CSVImageDataset(Dataset):

    def __init__(self, csv_file_path, root, image_size, tags, num_samples=0, no_val=False, val=False, ignore_tags=None, one_hot=True, dropna=True,
                 ind=1, limit_classes=False, augment=False, model=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        self.augment = augment
        if self.augment:
            self.aug_labels = torch.load("augmented_labels.pt")
            self.model = model
        if no_val:
            self.annotations = self.annotations[self.annotations.partition == "train"].reset_index(drop=True)
        elif val:
            dropna = False
            # num_samples = 0
            # tags = ["label"]
            # self.annotations = self.annotations[self.annotations.isnull().any(axis=1)]
        if dropna:
            self.annotations = self.annotations.dropna().reset_index(drop=True)

        self.annotations = self.annotations.sample(n=num_samples) if num_samples > 0 else self.annotations
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
            self.annotations.to_csv("unbias_annotations.csv", index=False)
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
        self.ind = ind
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        if self.augment:
            return len(self.aug_labels) + self.labels.shape[0]
        return self.labels.shape[0]

    def __label_len__(self):
        return self.labels.shape[1]

    def __all_possible_labels__(self):
        return torch.load(self.all_possible_labels)

    def __get_train_val_split__(self):
        train = self.annotations[self.annotations.partition == "train"].index.values
        val = self.annotations[self.annotations.partition == "val"].index.values
        return train, val

    def __get_labels__(self):
        if self.augment:
            return torch.cat((self.labels, self.aug_labels), dim=0)
        return self.labels

    def __getitem__(self, index):
        if index >= self.labels.shape[0]:
            label = self.aug_labels[index - self.labels.shape[0]].float().cuda()
            self.model.set_evaluation_parameters(reset=True, labels_to_evaluate=label.reshape(1, -1), total=1)
            average_generated_image = self.model.evaluate(use_mapper=True, only_ema=True, truncation_trick=1.)[0]
            return average_generated_image, label
        end = ".png" if self.ind == 0 else ""
        image_path = os.path.join(self.root, self.annotations.iloc[index, self.ind] + end)
        image = Image.open(image_path)

        labels = self.labels[index].cuda().float()
        if self.transform:
            image = self.transform(image).cuda()

        return image, labels


class CondGenDataset(Dataset):
    def __init__(self, model, num_samples, truncation_trick=1., labels=None, same_lats=False, save=False, use_mapper=True, image_size=None):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.truncation_trick = truncation_trick
        self.labels = labels
        self.use_mapper = use_mapper
        self.image_size = image_size
        self.lats = torch.randn(num_samples, model.latent_dim) if same_lats else None
        self.save = save
        self.num_digits = np.log10(num_samples).astype(int) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        label = self.labels[index].cuda().float()
        if self.lats is not None:
            lat = self.lats[index].cuda().float()
            self.model.set_evaluation_parameters(reset=True, labels_to_evaluate=label.reshape(1, -1),
                                                 latents_to_evaluate=lat.reshape(1, -1), total=1)
        else:
            self.model.set_evaluation_parameters(reset=True, labels_to_evaluate=label.reshape(1, -1), total=1)
        average_generated_image = self.model.evaluate(use_mapper=self.use_mapper, only_ema=True,
                                                      truncation_trick=self.truncation_trick)[0]
        if self.image_size != average_generated_image.shape[-1]:
            average_generated_image = transforms.Resize(self.image_size)(average_generated_image)
        if self.save:
            Image.fromarray(np.uint8(average_generated_image.cpu().permute(1, 2, 0).numpy()*255)).save(
                f"generated_images/{str(index).zfill(self.num_digits)}.png")
        return average_generated_image #, label[:7]


class GenDataset(Dataset):
    def __init__(self, model, num_samples=10000, truncation_trick=1.42, use_labels=False,
                 labels=None, use_mapper=True, image_size=None, save=False):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.truncation_trick = truncation_trick
        self.use_mapper = use_mapper
        self.image_size = image_size
        self.use_labels = use_labels
        self.labels = labels
        self.latents = [(torch.randn(num_samples, 512).cuda(), model.GAN.G.num_layers)]
        self.save = save
        self.num_digits = np.log10(num_samples).astype(int) + 1

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
        if self.save:
            Image.fromarray(np.uint8(average_generated_image.cpu().permute(1, 2, 0).numpy()*255)).save(
                f"generated_images/{str(idx).zfill(self.num_digits)}.png")
        return average_generated_image


class RealDataset(Dataset):
    def __init__(self, csv_file_path, root, image_size, num_samples=-1, use_labels=False, tags=None, dropna=True, ind=1):
        super().__init__()
        self.annotations = pd.read_csv(csv_file_path, sep=',')
        if dropna:
            self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.sample(n=num_samples) if num_samples > 0 else self.annotations
        self.root = root
        self.ind = ind
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
        image_path = os.path.join(self.root, self.annotations.iloc[index, self.ind])
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


def prepare_labels(num_samples, num_classes):
    labels = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        labels[i, i] = 1
    return labels
