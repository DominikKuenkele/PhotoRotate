import os
import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.models import ResNet101_Weights

ORIENTATIONS = (0, 90, 180, 270)
ORIENTATION_LABELS = {degree: index for index, degree in enumerate(ORIENTATIONS)}
ORIENTATION_INDEX = {index: degree for degree, index in ORIENTATION_LABELS.items()}


@dataclass
class Sample:
    image: Tensor
    label: Tensor
    path: str


class PhotoRotateDataset(Dataset):
    def __init__(
        self,
        photo_base_dir: str,
        number_photos_per_event_dir: int,
        max_samples: int,
        rotate: bool,
        earliest_year: int = 0,
        latest_year: int = 9999,
    ):
        super().__init__()

        self.samples: list[Sample] = []

        year_dirs = []
        all_years = os.listdir(photo_base_dir)
        for year_string in all_years:
            try:
                year = int(year_string)
                if year > earliest_year and year < latest_year:
                    year_dirs.append(os.path.join(photo_base_dir, year_string))
            except ValueError:
                continue

        number_event_dirs = int(max_samples / number_photos_per_event_dir)
        selected_dirs = self.select_random_dirs(
            year_dirs, number_event_dirs, number_photos_per_event_dir
        )

        for dir in selected_dirs:
            jpgs = [os.path.join(dir, file) for file in os.listdir(dir) if isJpeg(file)]
            selected_jpgs = random.sample(jpgs, number_photos_per_event_dir)
            for jpg in selected_jpgs:
                try:
                    image = Image.open(jpg).convert("RGB")
                except OSError:
                    continue

                orientaion_label = 0
                if rotate:
                    image = ImageOps.exif_transpose(image)

                    if image.size[0] > image.size[1]:
                        possible_orientations = (0, 180)
                    else:
                        possible_orientations = (90, 270)

                    selected_orientation = random.choice(possible_orientations)
                    orientaion_label = ORIENTATION_LABELS[selected_orientation]
                    image = image.rotate(selected_orientation)

                preprocessed_image = ResNet101_Weights.IMAGENET1K_V2.transforms()(image)
                self.samples.append(
                    Sample(
                        image=preprocessed_image,
                        label=torch.tensor(orientaion_label),
                        path=jpg,
                    )
                )
                if len(self.samples) % 20 == 0:
                    print(f"Loaded {len(self.samples)} images...", end="\r")

        print("Done loading images.               ")

    def select_random_dirs(
        self, year_dirs, num_event_dirs=10, number_photos_per_event_dir=50
    ):
        event_dirs = set()

        if len(year_dirs) != 0:
            while len(event_dirs) < num_event_dirs:
                print(
                    f"Selected {len(event_dirs)}/{num_event_dirs} random folders...",
                    end="\r",
                )

                year = random.choice(year_dirs)
                if os.path.isdir(year):
                    months = os.listdir(year)
                    if len(months) == 0:
                        continue
                    month_dir = os.path.join(year, random.choice(months))
                    if os.path.isdir(month_dir):
                        events = os.listdir(month_dir)
                        if len(events) == 0:
                            continue
                        event_dir = os.path.join(month_dir, random.choice(events))
                        if event_dir not in event_dirs and os.path.isdir(event_dir):
                            event_folders = [
                                folder
                                for folder in [
                                    event_dir,
                                    *[
                                        os.path.join(event_dir, dir)
                                        for dir in os.listdir(event_dir)
                                        if os.path.isdir(os.path.join(event_dir, dir))
                                    ],
                                ]
                                if hasJpegs(folder, number_photos_per_event_dir)
                            ]

                            if len(event_folders) == 0:
                                continue

                            event_dirs.add(random.choice(event_folders))

        print("Done selecting folders.               ")
        return list(event_dirs)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def hasJpegs(directory: str, threshold: int) -> bool:
    jpg_count = sum(1 for file in os.listdir(directory) if isJpeg(file))
    return jpg_count > threshold


def isJpeg(file: str) -> bool:
    return file.lower().endswith((".jpg", ".jpeg"))
