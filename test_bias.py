from utils import CSVImageDataset

from config import *

dataset = CSVImageDataset(csv_file_path=CSV_PATH, image_size=128, root=FOLDER, tags=TAGS,
                                       ignore_tags=IGNORE_TAGS, ind=1, no_val=True, limit_classes=True)
