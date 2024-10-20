"""The cats vs dogs dataset"""
from pathlib import Path
from typing import Callable

import lightning as L
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import utils


def download(download_dir: Path):
    # Download and extract the dataset to the default or specified directory
    utils.download_and_extract_zip(
        url="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
        save_path=download_dir.joinpath("cats_vs_dogs.zip"),
        extract_path=download_dir.joinpath("cats_vs_dogs/"),
    )

    # List of bad files to delete after extraction
    bad_files = [
        download_dir.joinpath("cats_vs_dogs") / "PetImages" / "Cat" / "666.jpg",
        download_dir.joinpath("cats_vs_dogs") / "PetImages" / "Dog" / "11702.jpg",
    ]

    # Delete bad files if they exist
    for bad_file in bad_files:
        utils.delete_bad_file(bad_file)


class CatsAndDogs(Dataset):
    """The Cats and Dogs Dataset."""

    def __init__(
        self,
        observations: list[dict],
        transform: Callable | None = None,
        classes: list[str] = None,
    ):
        """
        Args:
            observations: list[dict], where each dict must have "image_path" and "label"
            transform: Optional transform to be applied on an image
            classes: List of class names.
        """
        self.observations = observations
        self.transform = transform
        self.classes = (
            classes if classes is not None else sorted({x["label"] for x in observations})
        )

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx: int):
        image_path = self.observations[idx]["image_path"]
        image = Image.open(image_path)
        label = self.observations[idx]["label"]
        label_num = self.classes.index(label)

        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label_num}


class CatsAndDogsRandom(CatsAndDogs):
    """Modify parent class to return random image."""

    def __getitem__(self, idx: int):
        image_path = self.observations[idx]["image_path"]
        image = Image.open(image_path)
        label = self.observations[idx]["label"]
        label_num = self.classes.index(label)

        random_image = Image.fromarray(np.random.randint(0, 256, image.size, dtype=np.uint8))

        if self.transform:
            random_image = self.transform(random_image)
        return {"image": random_image, "label": label_num}


class CatsAndDogsData(L.LightningDataModule):
    """Create a data module to manage train, validation and test sets."""

    def __init__(
        self,
        train_observations: list[dict],
        val_observations: list[dict],
        test_observations: list[dict],
        classes: list[str],
        train_transform: Callable,
        test_transform: Callable,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_observations = train_observations
        self.val_observations = val_observations
        self.test_observations = test_observations

        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        """Split the dataset into train, validation, and test sets."""
        if stage == "fit" or stage is None:
            self.ds_train = CatsAndDogs(
                self.train_observations, transform=self.train_transform, classes=self.classes
            )
            self.ds_val = CatsAndDogs(
                self.val_observations, transform=self.test_transform, classes=self.classes
            )

        if stage == "test" or stage is None:
            self.ds_test = CatsAndDogs(
                self.test_observations, transform=self.test_transform, classes=self.classes
            )

    def train_dataloader(self):
        """Return the train data loader."""
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Return the validation data loader."""
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return the test data loader."""
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
