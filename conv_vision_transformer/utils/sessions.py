import os
import torch
from torchvision import transforms, datasets
from albumentations import HorizontalFlip, VerticalFlip, Affine, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, HueSaturationValue, GaussNoise, Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose
import numpy as np
from PIL import Image


def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf([GaussNoise()], p=0.2),
            Affine(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.2,
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p,
    )


def augment(aug, image):
    return aug(image=image)["image"]


class Aug:
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([Aug(), transforms.ToTensor(), transforms.Normalize(mean, std)]),
    "validation": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
    "test": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
}


def session(data_dir="sample/", batch_size=32):
    batch_size = batch_size
    data_dir = data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "validation", "test"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers=0, pin_memory=True) for x in ["train", "validation", "test"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation", "test"]}
    return batch_size, dataloaders, dataset_sizes
