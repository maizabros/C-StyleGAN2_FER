import torch
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import cycle
from utils import CSVImageDataset

# N_SAMPLES = 10000
BATCH_SIZE = 100
IMAGE_SIZE = 128
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"
getTrainValTest = lambda x, y, z: (x[:int(z*y[0])], x[int(z*y[0]):int(z*(y[0]+y[1]))], x[int(z*(y[0]+y[1])):])

dataset_reals = CSVImageDataset(csv_file_path=CSV_PATH, image_size=IMAGE_SIZE, num_samples=0, root=FOLDER, tags=["label"])
reals_indices = list(range(dataset_reals.__len__()))
np.random.seed(42)
np.random.shuffle(reals_indices)
train_r, val_r, test_r = getTrainValTest(reals_indices, [0.7, 0.15, 0.15], dataset_reals.__len__())
test_sampler = SubsetRandomSampler(test_r)

realsloader_test = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=BATCH_SIZE, drop_last=False,
                                    shuffle=False, pin_memory=False, sampler=test_sampler))

resnet = models.resnet50(pretrained=False).cuda()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 7)
resnet.load_state_dict(torch.load("models/test_all_2_100k/gen_model_2.pt"))
# resnet.load_state_dict(torch.load("models/test_all_4_100k/gen_model_0.pt"))


N_SAMPLES = len(test_r)
total = 0
progress = tqdm(range(N_SAMPLES // BATCH_SIZE))
for i in progress:
    images, labels = next(realsloader_test)
    resnet.cuda().eval()
    logits = resnet(images.cuda())
    total += torch.sum(logits.argmax(dim=1) == labels.argmax(dim=1)).item()
    if i % 10 == 0 and i > 0:
        progress.set_description(f"{100*total / (i * BATCH_SIZE)}%")
print(100*total/N_SAMPLES)
# torchvision.utils.save_image(images, "images.png", nrow=10)
