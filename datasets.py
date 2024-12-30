import abc
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


class ImageSelector:
    def __init__(
        self,
        base_dir: str,
        earliest_year: int,
        latest_year: int,
        number_photos_per_event_dir: int,
        *_args,
        **_kwargs,
    ):
        self.base_dir = base_dir
        self.earliest_year = earliest_year
        self.latest_year = latest_year
        self.number_photos_per_event_dir = number_photos_per_event_dir

    def _get_yer_dirs(self) -> list[str]:
        year_dirs = set()
        for year_string in os.listdir(self.base_dir):
            try:
                year = int(year_string)
                if self.earliest_year < year < self.latest_year:
                    year_dirs.add(os.path.join(self.base_dir, year_string))
            except ValueError:
                continue

        return list(year_dirs)

    def select_images(self, number_images=100):
        year_dirs = self._get_yer_dirs()

        if len(year_dirs) == 0:
            return []

        image_paths = set()
        selected_events = set()
        while len(image_paths) < number_images:
            print(
                f"Selecting {len(image_paths)}/{number_images} random images...",
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
                    if event_dir not in selected_events and os.path.isdir(event_dir):
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
                            if hasJpegs(folder, self.number_photos_per_event_dir)
                        ]

                        if len(event_folders) == 0:
                            continue

                        event = random.choice(event_folders)
                        selected_events.add(event)
                        jpgs = [
                            os.path.join(event, file)
                            for file in os.listdir(event)
                            if isJpeg(file)
                        ]
                        selected_jpgs = random.sample(
                            jpgs, self.number_photos_per_event_dir
                        )
                        image_paths.update(selected_jpgs)

        print("Done selecting images.               ")
        return list(image_paths)


@dataclass
class ProcessedImage:
    image: any
    label: str


class ImageProcessor(abc.ABC):
    def process(self, _image: Image.Image) -> ProcessedImage: ...


class TensorProcessor(ImageProcessor):
    def __init__(
        self,
        rotate: bool,
        *_args,
        **_kwargs,
    ):
        self.rotate = rotate

    def process(self, image: Image.Image) -> ProcessedImage:
        orientation_label = 0
        if self.rotate:
            image = ImageOps.exif_transpose(image)

            if image.size[0] > image.size[1]:
                possible_orientations = (0, 180)
            else:
                possible_orientations = (90, 270)

            selected_orientation = random.choice(possible_orientations)
            orientation_label = ORIENTATION_LABELS[selected_orientation]
            image = image.rotate(selected_orientation)

        longer_side = max(image.size[0], image.size[1])
        new_size = (longer_side, longer_side)
        new_im = Image.new("RGB", new_size)
        box = tuple((n - o) // 2 for n, o in zip(new_size, image.size))
        new_im.paste(image, box)

        preprocessed_image = ResNet101_Weights.IMAGENET1K_V2.transforms()(new_im)

        return ProcessedImage(
            image=preprocessed_image, label=torch.tensor(orientation_label)
        )


class DownscalingProcessor(ImageProcessor):
    def __init__(self, size: int):
        self.size = size

    def process(self, image: Image.Image):
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        return ProcessedImage(image=image, label=0)


@dataclass
class Sample:
    image: Tensor | Image.Image
    label: Tensor | int
    path: str


class PhotoRotateDataset(Dataset):
    def __init__(
        self,
        image_selector: ImageSelector,
        image_processor: ImageProcessor,
        max_samples: int,
        *_args,
        **_kwargs,
    ):
        super().__init__()

        self.samples: list[Sample] = []

        selected_images = image_selector.select_images(max_samples)

        for image_path in selected_images:
            try:
                image = Image.open(image_path).convert("RGB")
            except OSError:
                continue

            processed_image = image_processor.process(image)

            self.samples.append(
                Sample(
                    image=processed_image.image,
                    label=processed_image.label,
                    path=image_path,
                )
            )

            if len(self.samples) % 20 == 0:
                print(f"Processed {len(self.samples)} images...", end="\r")

        print("Done processing images.               ")

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def hasJpegs(directory: str, threshold: int) -> bool:
    jpg_count = sum(1 for file in os.listdir(directory) if isJpeg(file))
    return jpg_count > threshold


def isJpeg(file: str) -> bool:
    return file.lower().endswith((".jpg", ".jpeg"))
