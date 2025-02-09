import argparse
from test import MulticlassAccuracy, MulticlassConfusionMatrix

import torch
from attr import dataclass
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from callbacks import ConsoleLogger, FileLogger, ModelSaver
from datasets import PhotoRotateDatasetH5, Sample
from models import PhotoRotateAttentionModel, PhotoRotateModel, ResnetFeatureExtractor
from train import Trainer


@dataclass
class ModelConfiguration:
    dataset: torch.utils.data.Dataset
    model: nn.Module
    model_args: dict


CONFIGURATIONS: dict[str, ModelConfiguration] = {
    "simple": ModelConfiguration(
        dataset=PhotoRotateDatasetH5,
        model=PhotoRotateModel,
        model_args={
            "resnet": ResnetFeatureExtractor(
                pretrained=True,
                fine_tune=False,
                number_blocks=4,
                avgpool=False,
                fc=False,
            ),
        },
    ),
    "attention": ModelConfiguration(
        dataset=PhotoRotateDatasetH5,
        model=PhotoRotateAttentionModel,
        model_args={
            "resnet": ResnetFeatureExtractor(
                pretrained=True,
                fine_tune=False,
                number_blocks=4,
                avgpool=False,
                fc=False,
            ),
        },
    ),
}


def collate_fn(samples: list[Sample]):
    images = []
    labels = []
    for sample in samples:
        images.append(sample.image)
        labels.append(sample.label)

    return (torch.stack(images), torch.stack(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -- DATASET --
    parser.add_argument("--config", choices=CONFIGURATIONS.keys(), help="configuration")
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="Path to the extracted features",
    )

    # -- TRAINING --
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument(
        "--test_batch_size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory where to save the model",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="will be appended to the name of the model",
    )

    args = parser.parse_args()
    print(args)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        raise AttributeError("Device must be cpu or cuda")

    config = CONFIGURATIONS[args.config]

    dataset = config.dataset(args.dataset_file)
    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(
        dataset, (train_dataset_length, test_dataset_length)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )

    model_args = config.model_args
    model_args["dropout"] = args.dropout
    model = config.model(**model_args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_loader,
        test_data=test_loader,
        loss_function=loss_function,
        device=device,
        tester=[MulticlassAccuracy(device), MulticlassConfusionMatrix(4, device)],
        callbacks=[
            ConsoleLogger(),
            FileLogger(
                args.out_dir,
                [
                    args.config,
                    dataset.name,
                    len(dataset),
                    args.epochs,
                    args.lr,
                    args.dropout,
                    args.suffix,
                ],
            ),
            ModelSaver(
                args.out_dir,
                [
                    args.config,
                    dataset.name,
                    len(dataset),
                    args.epochs,
                    args.lr,
                    args.dropout,
                    args.suffix,
                ],
                as_state_dict=False,
            ),
        ],
    )

    trainer.train(args.epochs)
