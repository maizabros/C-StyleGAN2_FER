from torch import nn, Tensor, save
from torch.utils.data import DataLoader
from torchvision import models

from InceptionV3 import InceptionV3
from dataset import cycle
from metrics import compute_embeddings
from utils import RealDataset

root = '../models'
name = "test_all_4_100k"

vgg = models.vgg16(pretrained=True)
vgg.classifier = nn.Sequential(*[vgg.classifier[i] for i in range(5)])
vgg = vgg.cuda().eval()

inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()

N_SAMPLES = 0
BATCH_SIZE = 50
IMAGE_SIZE = 128
STEP = 1
N_STATS = 1
TRUNC = 1
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"

dataset_reals = RealDataset(CSV_PATH, FOLDER, IMAGE_SIZE, num_samples=N_SAMPLES)
realsloader = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=BATCH_SIZE, drop_last=True, shuffle=False,
                               pin_memory=False))
count = dataset_reals.__len__() // BATCH_SIZE
embeds_inc = compute_embeddings(realsloader, count, BATCH_SIZE, inc, 2048, "FID", verbose=True)
embeds_vgg = compute_embeddings(realsloader, count, BATCH_SIZE, vgg, 4096, "PR", verbose=True)

save(Tensor(embeds_inc), FOLDER + "\\real_embeddings_inception.pt")
save(Tensor(embeds_vgg), FOLDER + "\\real_embeddings_vgg_pr_re.pt")
