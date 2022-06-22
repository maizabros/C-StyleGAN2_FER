import os
import json
import fire
import torch

from trainer import Trainer
import torchvision


root = 'models'
NAME = "test_all_3_100k"

IMAGES_TO_GENERATE = 10
ROWS = 2
COLS = 5


def generate(name=NAME, images_to_generate=IMAGES_TO_GENERATE, use_mapper=True, truncation_trick=1.):
    with open(os.path.join(root, name, 'config.json'), 'r') as file:
        config = json.load(file)
    model = Trainer(**config)
    model.load(-1, root=root)
    """
        0    |      1     |    2   |        3        |        4       |        5        |    6     |   7   |  8  
     --------|------------|--------|-----------------|----------------|-----------------|----------|-------|----- 
      angry  | disgust    | fear   | happy           | neutral        | sad             | surprise |       |     
      0-2    | 10-19      | 20-29  | 3-9             | 30-39          | 40-49           |  50-59   | 60-69 | 70+ 
      Female | Male       |        |                 |                |                 |          |       |     
      Black  | East Asian | Indian | Latino Hispanic | Middle Eastern | Southeast Asian | White    |       |     
      Asian  | Black      | Indian | White           |                |                 |          |       |     

    """"""                 0  1  2  3  4  5  6  7  8   """
    label1 = torch.Tensor([0, 0, 0, 1, 0, 0, 0,        # Expression
                           0, 0, 1, 0, 0, 0, 0, 0, 0,  # Age
                           0, 1,                       # Sex
                           0, 1, 0, 0, 0, 0, 0,        # Race7
                           1, 0, 0, 0])                # Race4
    labels = label1.repeat(ROWS*COLS, 1)
    test_image = None
    for i in range(images_to_generate):
        model.set_evaluation_parameters(labels_to_evaluate=labels.cuda(), reset=True, total=ROWS*COLS)
        generated_images, average_generated_images = model.evaluate(use_mapper=use_mapper,
                                                                    truncation_trick=truncation_trick)

        test_image = torch.cat((test_image, average_generated_images)) if i != 0 else average_generated_images
        # for j, im in enumerate(average_generated_images):
        #     torchvision.utils.save_image(im, f'test\{i}-{j}.png', nrow=model.label_dim)

    torchvision.utils.save_image(test_image, f'test.png', nrow=ROWS*COLS)


if __name__ == "__main__":
    fire.Fire(generate)
