import argparse
import pickle

from datasets import PhotoRotateDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        help="Path to the base directory of all datasets",
    )
    parser.add_argument("--earliest_year", type=int, default=0, help="start dir")
    parser.add_argument("--latest_year", type=int, default=9999, help="start dir")
    parser.add_argument(
        "--number_photos_per_subdir", type=int, default=50, help="start dir"
    )
    parser.add_argument(
        "--max_samples", type=int, default=10_000, help="max samples to load"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to the save file",
    )

    args = parser.parse_args()
    print(args)

    dataset = PhotoRotateDataset(
        args.dataset_base_dir,
        args.number_photos_per_subdir,
        args.max_samples,
        True,
        args.earliest_year,
        args.latest_year,
    )

    with open(args.out_file, "wb") as f:
        pickle.dump(dataset, f)
