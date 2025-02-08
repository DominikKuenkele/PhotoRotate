import argparse
import os
import pickle
from dataclasses import dataclass

from datasets import (
    DatasetImageSelector,
    DownscalingProcessor,
    ImageProcessor,
    ImageSelector,
    PhotoRotateDataset,
    TensorProcessor,
)


@dataclass
class Dataset:
    image_selector: ImageSelector
    image_processor: ImageProcessor
    image_processor_args: dict


CONFIGURATIONS = {
    "tensor": Dataset(
        image_selector=ImageSelector,
        image_processor=TensorProcessor,
        image_processor_args={"rotate": True},
    ),
    "downscaling": Dataset(
        image_selector=ImageSelector,
        image_processor=DownscalingProcessor,
        image_processor_args={"size": 256},
    ),
    "dataset": Dataset(
        image_selector=DatasetImageSelector,
        image_processor=TensorProcessor,
        image_processor_args={"rotate": True},
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        help="Path to the base directory of all datasets",
    )
    parser.add_argument("--earliest_year", type=int, default=0, help="start dir")
    parser.add_argument("--latest_year", type=int, default=9999, help="last dir")
    parser.add_argument(
        "--number_photos_per_event",
        type=int,
        default=50,
        help="number of photos per event",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="file to pickle PhotoRotateDataset",
    )
    parser.add_argument(
        "--max_samples", type=int, default=10_000, help="max samples to load"
    )
    parser.add_argument(
        "--configuration", choices=CONFIGURATIONS.keys(), help="dataset configuration"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to the out directory",
    )

    args = parser.parse_args()
    print(args)

    dataset_configuration = CONFIGURATIONS[args.configuration]

    image_selector_args = {
        "base_dir": args.dataset_base_dir,
        "earliest_year": args.earliest_year,
        "latest_year": args.latest_year,
        "number_photos_per_event_dir": args.number_photos_per_event,
        "dataset_file": args.dataset_file,
    }
    image_selector = dataset_configuration.image_selector(**image_selector_args)

    image_processor_args = {}
    image_processor_args |= dataset_configuration.image_processor_args
    image_processor = dataset_configuration.image_processor(**image_processor_args)

    dataset = PhotoRotateDataset(
        image_selector=image_selector,
        image_processor=image_processor,
        max_samples=args.max_samples,
    )

    print("Saving...", end="\r")
    file_name = f"dataset_{args.configuration}_{len(dataset)}.pickle"
    file = os.path.join(args.out_dir, file_name)
    with open(file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved to {file_name}")
