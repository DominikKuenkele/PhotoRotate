import argparse
import pickle
from test import test

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import Mean

from datasets import Sample
from models import PhotoRotateModel, ResnetFeatureExtractor


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
    parser.add_argument(
        "--test_batch_size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to the save file",
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
        )
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    print(f"Batches per epoch: {len(train_loader)}")
    for epoch in range(args.epochs):
        total_loss = Mean(device=device)
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            model_input = inputs.to(device)
            ground_truth = labels.to(device)

            output = model(model_input)

            loss = loss_function(output, ground_truth)
            total_loss.update(loss)

            loss_string = f"epoch {epoch}, batch {i}: {total_loss.compute():.4f}"
            print(loss_string, end="\r")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()
        accuracy = test(model, test_loader, device)
        print(f"Accuracy: {accuracy.compute()}")

    torch.save(model.state_dict(), args.out_file)
