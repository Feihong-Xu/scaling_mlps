import argparse
import os

import numpy as np
import torch
import torchvision
from ffcv.fields import BytesField, IntField, RGBImageField
from ffcv.writer import DatasetWriter

class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, data_dir, corruption, stage='test', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.corruption = corruption
        self.stage = stage
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = np.load(os.path.join(data_dir, f'{corruption}.npy'))
        self.img_labels = np.load(os.path.join(data_dir, 'labels.npy'))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image, label


def get_dataset(dataset_name, mode, data_path, oo=None):
    if data_path is not None:
        return torchvision.datasets.ImageFolder(root=data_path, transform=None)

    if dataset_name == "cifar10":
        if mode in ["train", "test"]:
            return torchvision.datasets.CIFAR10(
                root="/tmp", train=mode == "train", download=True
            )
        elif mode == "ood":
            return CIFAR10C(
                data_dir="tmp/CIFAR10C",
                corruption=oo,
                stage="test",
                transform=None,
                target_transform=None,
            )

    elif dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root="/tmp", train=mode == "train", download=True
        )
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} not supported. Please add it here."
        )


def create_beton(args):
    if args.mode == "ood":
        oos = [i.split("/")[-1].split(".")[0] for i in os.listdir("tmp/CIFAR10C") if i.split("/")[-1].split(".")[0] != "labels"]
        for oo in oos:
            dataset = get_dataset(args.dataset_name, args.mode, args.data_path, oo)
            write_path = os.path.join(
                args.write_path, args.dataset_name, args.mode, oo, f"{args.mode}_{oo}_{args.res}.beton"
            )
            os.makedirs(os.path.dirname(write_path), exist_ok=True)

            writer = DatasetWriter(
                write_path,
                {
                    "image": RGBImageField(write_mode="smart", max_resolution=args.res),
                    "label": IntField(),
                },
                num_workers=args.num_workers,
            )

            writer.from_indexed_dataset(dataset, chunksize=100)
    else:
        write_path = os.path.join(
            args.write_path, args.dataset_name, args.mode, f"{args.mode}_{args.res}.beton"
        )

        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        writer = DatasetWriter(
            write_path,
            {
                "image": RGBImageField(write_mode="smart", max_resolution=args.res),
                "label": IntField(),
            },
            num_workers=args.num_workers,
        )

        writer.from_indexed_dataset(dataset, chunksize=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to .beton format")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="path to dataset if data is given in a hierarchical subfolder structure.",
    )
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--res", type=int, default=32, help="resolution of images")
    parser.add_argument(
        "--write_path",
        type=str,
        default="./beton/",
        help="path to write .beton file to",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="number of workers to use"
    )
    args = parser.parse_args()

    assert (
        args.dataset_name is not None or args.data_path is not None
    ), "Either dataset_name or data_path must be specified."

    create_beton(args)
