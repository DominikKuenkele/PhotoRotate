from torch.utils.data import Dataset
import os
from dataclasses import dataclass
from torch import Tensor
import torch
import random
from PIL import Image
from torchvision.models import ResNet101_Weights

@dataclass
class Sample:
    image: Tensor
    label: Tensor


class PhotoRotateDataset(Dataset):
    def __init__(self, photo_base_dir: str, earliest_year: int, number_photos_per_subdir: int, max_samples: int):
        super().__init__()

        self.samples: list[Sample] = []


        year_dirs = []
        all_years = os.listdir(photo_base_dir)
        for year_string in all_years:
            try:
                year = int(year_string)
                if year > earliest_year:
                    year_dirs.append(os.path.join(photo_base_dir, year_string))
            except ValueError:
                continue


        number_dirs = int(max_samples / number_photos_per_subdir)
        selected_dirs = self.select_random_dirs(year_dirs, number_dirs, number_photos_per_subdir)

        for dir in selected_dirs:
            jpgs = [os.path.join(dir, file) for file in os.listdir(dir) if isJpeg(file)]
            selected_jpgs = random.sample(jpgs, number_photos_per_subdir)
            for jpg in selected_jpgs:
                try:
                    image = Image.open(jpg).convert("RGB")
                except OSError:
                    continue

                preprocessed_image = ResNet101_Weights.IMAGENET1K_V2.transforms()(image)
                self.samples.append(Sample(
                    image=preprocessed_image,
                    label=torch.tensor(int(image.size[0]>image.size[1]))
                ))
                if len(self.samples) % 20 == 0:
                    print(f"Loaded {len(self.samples)} images...", end="\r")

        print("Done loading images.")


    def select_random_dirs(self, years_dirs, num_subsubdirs=10, number_photos_per_subdir=50):
        dirs = set()

        while len(dirs) < num_subsubdirs:
            year = random.choice(years_dirs)
            if os.path.isdir(year):
                month = random.choice(os.listdir(year))
                month_dir = os.path.join(year, month)
                if month_dir not in dirs and os.path.isdir(month_dir):
                    jpg_count = sum(1 for file in os.listdir(month_dir) if isJpeg(file))
                    if jpg_count > number_photos_per_subdir:
                        dirs.add(month_dir)

        return list(dirs)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def isJpeg(file: str) -> bool:
    return file.lower().endswith(('.jpg', '.jpeg'))