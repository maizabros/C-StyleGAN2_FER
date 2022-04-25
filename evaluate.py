import os
import json
import fire
import torch

from trainer import Trainer
import torchvision

root = 'models'
NAME = "test_all_1_100k"

IMAGES_TO_GENERATE = 20


def generate(name=NAME, images_to_generate=IMAGES_TO_GENERATE, use_mapper=True, truncation_trick=1.):
    with open(os.path.join(root, name, 'config.json'), 'r') as file:
        config = json.load(file)
    model = Trainer(**config)
    model.load(200, root=root)

    label1 = [0, 1, 1,     # Happy
              0, 0, 1, 0,  # 20-29
              1,           # Male
              0, 1, 0, 1,  # South-Asian
              0, 0, 0]     # Asian

    for i in range(images_to_generate):
        model.set_evaluation_parameters(labels_to_evaluate=torch.Tensor([label1]).cuda(), reset=True)
        generated_images, average_generated_images = model.evaluate(use_mapper=use_mapper,
                                                                    truncation_trick=truncation_trick)

        torchvision.utils.save_image(average_generated_images, 'test.png', nrow=model.label_dim)

        for j, im in enumerate(average_generated_images):
            torchvision.utils.save_image(im, f'test\{i}-{j}.png', nrow=model.label_dim)


if __name__ == "__main__":
    fire.Fire(generate)
