import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


root_dir = "/teamspace/studios/this_studio/SKU110K_fixed/"


def load_sku_dataset(split):
    df = pd.read_csv(f"{root_dir}annotations/annotations_{split}.csv", header=None)
    columns = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height" ]
    df.columns = columns
    sku_110k_dataset = []

    for _, g in df.groupby(["image_name", "class"]):
        image_path = f"{root_dir}images/{g.iloc[0]['image_name']}"
        image = Image.open(image_path)
        img_width, img_height = image.size
        objects = []
        for idx, row in g.iterrows():
            x1, y1, x2, y2 = row[1:5]
            label = row[5]
            # normalize to 0-1
            x1 = x1 / img_width
            y1 = y1 / img_height
            x2 = x2 / img_width
            y2 = y2 / img_height

            objects.append({'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2 })

        sku_110k_dataset.append({'image_name': g.iloc[0]['image_name'], 'image': image, 'objects': objects, 'label': label})

    return sku_110k_dataset


class SKUDetectionDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # from the dataset  get image, convert to xmin, ymin, xmax, ymax
        sample = self.dataset[idx]
        image = sample["image"]
        objects = sample["objects"]
        label = sample["label"]

        # Extract the image, query, and bounding boxes
        return image, label, objects



def load_object_detection_dataset(split):

    dataset = load_sku_dataset(split)
    return SKUDetectionDataset(dataset)
