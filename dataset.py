import albumentations as A
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.labels[idx]

        if self.transform is None:
            self.transform = A.Compose([
                # height=40, width=150 for horizontal captcha
                A.Resize(40, 150),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])

        image = np.array(image)
        transformed = self.transform(image=image)
        image = transformed["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long)
        }
