import torch
from PIL import Image
import pandas as pd
import os
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

image_path = './img_locations.txt'
annotations_path = './annotations.csv'


def collate_fn(batch):
    """Makes a batch"""
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = default_collate([i for i in items[2] if torch.is_tensor(i)])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    return items


class OIDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def _get_img_name(self, path):
        """Converts path to image to image id"""
        return os.path.splitext(os.path.split(path)[1])[0]

    def __len__(self):
        df = pd.read_csv(annotations_path)
        return len(df)

    def __getitem__(self, item):
        """Gets image and annotation at a given index"""
        image = None
        with open(image_path) as f_i:
            for i, line in enumerate(f_i):
                if i == item:
                    path_to_img = str.rstrip(line)
                    image = Image.open(path_to_img).convert("RGB")

        if image is None:
            return None, None

        df = pd.read_csv(annotations_path)
        line = df.loc[df['imgid'] == self._get_img_name(path_to_img)].to_numpy()
        label_id = torch.tensor([line[0][5] + 1])  # 1 2 3 instead of 0 1 2 (0 is background)
        bbox = torch.tensor([[line[0][1], line[0][2], line[0][3], line[0][4]]])
        if self.transform is not None:
            image, bbox, label_id = self.transform(image, bbox, label_id)
        # opened image, id, bbox, label id
        return image, line[0][0], bbox, label_id
