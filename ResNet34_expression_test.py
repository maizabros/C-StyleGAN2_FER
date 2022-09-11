import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from trainer import Trainer
from sklearn.metrics import classification_report, accuracy_score
import json
from utils import CondGenDataset, CSVImageDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_optimizer import DiffGrad
from torch.optim import Adam
from dataset import cycle
import os
from tqdm import tqdm
from config import TAGS


N_EPOCHS = 15

root = "final_models"
name = "test_all_4_100k"

N_MODEL = 557
# N_SAMPLES = 100000

BATCH_SIZE = 50
IMAGE_SIZE = 256
TRUNC = 1.25
FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"
CSV_PATH = FOLDER + "\\affectnet_complete.csv"
# Tensorboard stuff
EXP_NAME = "logs_augmented_256_good_sampler"
# log_real = SummaryWriter(f"LOGS/{EXP_NAME}/real")
log_gen = SummaryWriter(f"resnet_models/{EXP_NAME}/gen")
step = 0
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def get_metrics(pred, targ):
    return classification_report(pred.argmax(dim=1).cpu(), targ.argmax(dim=1).cpu(), labels=range(7),
                                 target_names=CLASS_NAMES, zero_division=0, output_dict=True)


getTrainTest = lambda x, y, z: (x[:int(z*y[0])], x[int(z*y[0]):])

with open(os.path.join(root, name, 'config.json'), 'r') as f:
    config = json.load(f)

model = Trainer(**config)
model.load(N_MODEL, root=root)
#
dataset_reals = CSVImageDataset(csv_file_path=CSV_PATH, image_size=IMAGE_SIZE, num_samples=0, root=FOLDER, tags=TAGS,
                                augment=True, model=model)
# # reals_indices = list(range(dataset_reals.__len__()))
train_r, test_r = dataset_reals.__get_train_val_split__()

train_r = np.concatenate((train_r, np.arange(max(train_r.max(), test_r.max()) +1, max(train_r.max(), test_r.max()) +1 + torch.load("augmented_labels.pt").shape[0])))
len_train_r = len(train_r)
# # np.random.seed(42)
# # np.random.shuffle(reals_indices)
# # train_r, test_r = getTrainTest(reals_indices, [0.8, 0.2], dataset_reals.__len__())
train_sampler, test_sampler = SubsetRandomSampler(train_r), SubsetRandomSampler(test_r)
# realsloader = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=BATCH_SIZE,
#                                drop_last=False, shuffle=False, pin_memory=False, sampler=train_sampler))
#
realsloader_test = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=BATCH_SIZE,
                                    drop_last=False, pin_memory=False, sampler=test_sampler))
#
# N_SAMPLES = len(train_r)
# # labels = torch.load("all_possible_labels.pt")
# labels = torch.unique(dataset_reals.__get_labels__(), dim=0)
# print(" AFFECT | Número de expresiones: ", torch.unique(dataset_reals.__get_labels__()[:, :7], dim=0, return_counts=True)[1])
# print("  ANTES | Número de expresiones: ", labels[:, :7].unique(dim=0, return_counts=True)[1])
# print("  ANTES | Número de etiquetas diferentes: ", len(labels.unique(dim=0)))
# N_REPEAT = N_SAMPLES // len(labels) + 1
# labels = labels.repeat(N_REPEAT, 1)[torch.randperm(N_SAMPLES)]
#
# # labels = dataset_reals.__get_labels__()
# dataset_generator = CondGenDataset(model, num_samples=N_SAMPLES, labels=labels, truncation_trick=TRUNC,
#                                    use_mapper=True, image_size=IMAGE_SIZE)

genloader = cycle(DataLoader(dataset_reals, num_workers=0, batch_size=BATCH_SIZE,
                             drop_last=False, pin_memory=False, sampler=train_sampler))
labels = dataset_reals.__get_labels__()
print("Reals:", len_train_r, " Test: ", len(test_r))
# print("Generated:", N_SAMPLES)

print("DESPUÉS | Número de expresiones: ", labels[:, :7].unique(dim=0, return_counts=True)[1])
print("DESPUÉS | Número de etiquetas diferentes: ", len(labels.unique(dim=0)))


# resnet_reals = models.resnet50(pretrained=False).cuda()
resnet_gen = models.resnet50(pretrained=False).cuda()

# resnet_reals.fc = torch.nn.Linear(resnet_reals.fc.in_features, 7)
resnet_gen.fc = torch.nn.Linear(resnet_gen.fc.in_features, 7)

# reals_opt = DiffGrad(resnet_reals.parameters(), lr=0.001, betas=(0.5, 0.9))
gen_opt = DiffGrad(resnet_gen.parameters(), lr=0.001, betas=(0.95, 0.99), eps=1e-8)

for epoch in range(N_EPOCHS):
    progress = tqdm(range(len_train_r // BATCH_SIZE), total=len_train_r // BATCH_SIZE)
    if epoch == 0:
        desc = f"Epoch: {epoch}, Real Loss: {0.0}, Gen Loss: {0.0}" + \
               f" || Reals | acc: {0.0}, pr: {0.0}, rec: {0.0}, f1: {0.0}" + \
               f" || Gener | acc: {0.0}, pr: {0.0}, rec: {0.0}, f1: {0.0}"
        progress.set_description(desc)

    for i in progress:
        # real_images, real_labels = next(iter(realsloader))
        gen_images, gen_labels = next(iter(genloader))
        # real_images = real_images.cuda()
        # real_labels = real_labels[:, :7].cuda()
        gen_images = gen_images.cuda()
        gen_labels = gen_labels[:, :7].cuda()
        # resnet_reals.cuda().train()
        # reals_opt.zero_grad()
        resnet_gen.cuda().train()
        gen_opt.zero_grad()

        # real_logits = resnet_reals(real_images)
        # real_loss = F.cross_entropy(real_logits, real_labels)
        # real_loss.backward()
        # reals_opt.step()

        gen_logits = resnet_gen(gen_images)
        gen_loss = F.cross_entropy(gen_logits, gen_labels)
        gen_loss.backward()
        gen_opt.step()

        if i % 10 == 0:
            # real_report = get_metrics(real_logits, real_labels)
            gen_report = get_metrics(gen_logits, gen_labels)
            # real_acc = accuracy_score(real_labels.argmax(dim=1).cpu().numpy(), real_logits.argmax(dim=1).cpu().numpy())
            gen_acc = accuracy_score(gen_labels.argmax(dim=1).cpu().numpy(), gen_logits.argmax(dim=1).cpu().numpy())
            # real_rec, real_pr, real_f1, _ = real_report["macro avg"].values()
            gen_rec, gen_pr, gen_f1, _ = gen_report["macro avg"].values()
            desc = f"Epoch: {epoch}, Gen Loss: {gen_loss.item():.4f}" + \
                   f" || Gener | acc: {gen_acc:.4f}, pr: {gen_pr:.4f}, rec: {gen_rec:.4f}, f1: {gen_f1:.4f}"
            # "Real Loss: {real_loss.item():.4f}, "
 #            f" || Reals | acc: {real_acc:.4f}, pr: {real_pr:.4f}, rec: {real_rec:.4f}, f1: {real_f1:.4f}" + \
 # \
            progress.set_description(desc)
            # progress.set_description("Epoch: {}, Loss: {}, Acc: {}".format(epoch, gen_loss.item(), gen_acc.item()))
            progress.refresh()
            # log_real.add_scalar("Loss reales", real_loss.item(), global_step=step)
            # log_real.add_scalars("Metrics reales",
            #                      {"acc": real_acc, "pr": real_pr, "rec": real_rec, "f1": real_f1}, global_step=step)
            log_gen.add_scalar("Loss generados", gen_loss.item(), global_step=step)
            log_gen.add_scalars("Metrics generados",
                                {"acc": gen_acc, "pr": gen_pr, "rec": gen_rec, "f1": gen_f1}, global_step=step)

        if i == len_train_r // BATCH_SIZE - 1:
            # real_test = np.zeros(4, dtype=np.float32)
            gen_test = np.zeros(4, dtype=np.float32)
            key = "accuracy"
            for j in range(len(test_r) // BATCH_SIZE):
                with torch.no_grad():
                    # resnet_reals.cuda().eval()
                    resnet_gen.cuda().eval()

                    test_images, test_labels = next(iter(realsloader_test))
                    # test_logits = resnet_reals(test_images.cuda())
                    test_labels = test_labels[:, :7]
                    # test_report = get_metrics(test_logits, test_labels)
                    # real_test[0] += accuracy_score(
                    #     test_labels.argmax(dim=1).cpu().numpy(), test_logits.argmax(dim=1).cpu().numpy()
                    # ) / (len(test_r) // BATCH_SIZE)
                    # real_test[1:] += np.array(list(test_report["macro avg"].values())[:3]) / (len(test_r) // BATCH_SIZE)
                    test_logits = resnet_gen(test_images.cuda())
                    test_report = get_metrics(test_logits, test_labels)
                    gen_test[0] += accuracy_score(
                        test_labels.argmax(dim=1).cpu().numpy(), test_logits.argmax(dim=1).cpu().numpy()
                    ) / (len(test_r) // BATCH_SIZE)
                    gen_test[1:] += np.array(list(test_report["macro avg"].values())[:3]) / (len(test_r) // BATCH_SIZE)

                    progress.set_description(f"TESTING: {100 * j / (len(test_r) // BATCH_SIZE - 1):.2f}%")
            progress.set_description(desc)
            # log_real.add_scalars("Metrics reales test",
            #                      {"acc": real_test[0], "pr": real_test[1], "rec": real_test[2], "f1": real_test[3]},
            #                      global_step=step)
            log_gen.add_scalars("Metrics generados test",
                                {"acc": gen_test[0], "pr": gen_test[1], "rec": gen_test[2], "f1": gen_test[3]},
                                global_step=step)

            torch.save({"model": resnet_gen.state_dict(), "opt": gen_opt.state_dict()},
                       os.path.join("resnet_models", EXP_NAME, "gen_model_{}.pt".format(epoch)))
            # torch.save({"model": resnet_reals.state_dict(), "opt": reals_opt.state_dict()},
            #            os.path.join("resnet_models", EXP_NAME, "real_model_{}.pt".format(epoch)))
        step += 1
