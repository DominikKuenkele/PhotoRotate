import argparse
import pickle
from test import MulticlassAccuracy, MulticlassConfusionMatrix

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from callbacks import ConsoleLogger, FileLogger, ModelSaver
from datasets import Sample
from models import PhotoRotateModel, ResnetFeatureExtractor
from train import Trainer


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

    with open(args.dataset_file, "rb") as f:
        dataset = pickle.load(f)
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

    model = PhotoRotateModel(
        ResnetFeatureExtractor(
            pretrained=True,
            fine_tune=False,
            number_blocks=4,
            avgpool=False,
            fc=False,
        ),
        args.dropout,
    ).to(device)

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
                [len(dataset), args.epochs, args.lr, args.dropout, args.suffix],
            ),
            ModelSaver(
                args.out_dir,
                [len(dataset), args.epochs, args.lr, args.dropout, args.suffix],
            ),
        ],
    )

    trainer.train(args.epochs)
