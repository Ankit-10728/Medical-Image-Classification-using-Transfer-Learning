
from pathlib import Path
import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMG_DIR = Path("data/raw/images")
SPLITS_PATH = Path("data/splits.csv")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Pretrained models learned on ImageNet photos normalized with these values,
# so our images must be normalized the same way.
# advantage of normalization is that 
# since the model was trained over the same normalization techniqe so it is better to train on same scale 
# after normalization the gradients are smalll so it takes less time to (epochs) to train as the zig-zag curve with high gradient would take more time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(split, img_size=224):
    """Training images get random changes (augmentation); val/test images don't."""
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class SkinLesionDataset(Dataset):
    """Loads the images of one split ("train", "val" or "test")."""

    def __init__(self, split, img_size=224, splits_path=SPLITS_PATH, img_dir=IMG_DIR):
        df = pd.read_csv(splits_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = get_transforms(split, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
        image = self.transform(image=np.array(image))["image"]
        label = CLASS_TO_IDX[row["dx"]]
        return image, label


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = SkinLesionDataset(split)
        img, label = ds[0]
        print(f"{split}: {len(ds)} images | sample shape {tuple(img.shape)} "
              f"| label {label} ({CLASSES[label]})")