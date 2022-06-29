import torch
import torchvision.models as models
from trainer import Trainer
import json
import os

root = "models"
name = "test_all_2_100k"

with open('config.json') as f:
    config = json.load(f)

model = Trainer(**config)
model.load(705, root=root)

resnet = models.resnet34(pretrained=False)

# Change the last layer to logits vector of size 7
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 7)

# Change image size for first layer to 128x128


