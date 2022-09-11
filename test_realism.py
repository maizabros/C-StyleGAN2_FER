import json
import os

from metrics import *
from dataset import cycle
from trainer import Trainer
from utils import CondGenDataset

root = 'final_models'
name = "test_all_2_100k"

N_SAMPLES = 10
N_DIGITS = np.log10(N_SAMPLES).astype(int) + 1
BATCH_SIZE = 5
IMAGE_SIZE = 128
STEP = 1
N_STATS = 1
TRUNC = 1
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"
REAL_EMBEDS = {"FID": FOLDER + "\\real_embeddings_inception.pt", "PR": FOLDER + "\\real_embeddings_vgg_pr_re.pt",
               "Realism": FOLDER + "\\real_embeddings_vgg_pr_re.pt"}

f1_score = lambda x, y: (2 * x * y) / (x + y) if (x+y) != 0 else 0.0

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)

m = -1

embedding_model = models.vgg16(pretrained=True)
embedding_model.classifier = torch.nn.Sequential(*[embedding_model.classifier[i] for i in range(5)])
embedding_model = embedding_model.cuda().eval()

save = True
num_features = 4096

model.load(m, root=root)
labels = torch.load("all_possible_labels.pt")[:N_SAMPLES]
# labels = torch.unique(dataset_reals.__get_labels__(), dim=0)
# N_REPEAT = N_SAMPLES // len(labels) + 1
# labels = labels.repeat(N_REPEAT, 1)[torch.randperm(N_SAMPLES)]
dataset_generator = CondGenDataset(model, num_samples=N_SAMPLES, labels=labels, truncation_trick=TRUNC,
                                   use_mapper=True, image_size=IMAGE_SIZE, save=True)
print("Datasets loaded")
genloader = cycle(DataLoader(dataset_generator, num_workers=0, batch_size=BATCH_SIZE,
                             drop_last=False, shuffle=False, pin_memory=False))

real_embeds = torch.load(REAL_EMBEDS["Realism"])[torch.randperm(286300)[:N_SAMPLES]]
print("Real embeddings loaded")
gen_embeds = compute_embeddings(genloader, N_SAMPLES, BATCH_SIZE, embedding_model, num_features, "Realism")
print("Generator embeddings computed")

print("Calculating Realism....", end="\r")
values = realism(real_embeds, gen_embeds)
print("Calculating Realism.... Done")


print(values.mean())
print(len(values))
for i, val in tqdm(enumerate(values)):
    os.rename(f"generated_images\\{str(i).zfill(N_DIGITS)}.png", f"generated_images\\{val}_{str(i).zfill(N_DIGITS)}.png")
