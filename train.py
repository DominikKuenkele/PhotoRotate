import argparse
from models import PhotoRotateModel
from datasets import PhotoRotateDataset, Sample
from test import test

import os
import torch

from torcheval.metrics import Mean

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

def collate_fn(samples: list[Sample]):
    images = []
    labels = []
    for sample in samples:
        images.append(sample.image)
        labels.append(sample.label)
    
    return (
        torch.stack(images),
        torch.stack(labels)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -- DATASET --
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        help="Path to the base directory of all datasets",
    )
    parser.add_argument(
        "--earliest_year", type=int, default=0, help="start dir"
    )
    parser.add_argument(
        "--number_photos_per_subdir", type=int, default=50, help="start dir"
    )
    parser.add_argument(
        "--max_samples", type=int, default=10_000, help="max samples to load"
    )

    # -- TRAINING --
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    args = parser.parse_args()
    print(args)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        raise AttributeError("Device must be cpu or cuda")


    dataset = PhotoRotateDataset(args.dataset_base_dir, args.earliest_year, args.number_photos_per_subdir, args.max_samples)
    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(
        dataset, (train_dataset_length, test_dataset_length)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    model = PhotoRotateModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCEWithLogitsLoss()

    print(f"Batches per epoch: {len(train_loader)}")
    for epoch in range(args.epochs):
        total_loss = Mean(device=device)
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            model_input = inputs.to(device)
            ground_truth = labels.to(device).unsqueeze(1)

            output = model(model_input)

            loss = loss_function(output, ground_truth.float())
            total_loss.update(loss)

            loss_string = f"epoch {epoch}, batch {i}: {total_loss.compute():.4f}"
            print(loss_string, end="\r")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()
        accuracy = test(model, test_loader, device)
        print(f"Accuracy: {accuracy.compute()}")

    torch.save(model.state_dict(), os.path.join("out", "model.pth"))