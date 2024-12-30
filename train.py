from test import Tester
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torcheval.metrics import Mean

from callbacks import Callback


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_data: DataLoader,
        test_data: DataLoader,
        loss_function: Callable,
        device: torch.device,
        tester: list[Tester],
        callbacks: list[Callback],
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.loss_function = loss_function
        self.device = device
        self.tester = tester
        self.callbacks = callbacks

    def train(self, epochs: int):
        for callback in self.callbacks:
            callback.on_train_begin(self.model, self.train_data)

        for epoch in range(epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            loss = self.train_epoch()

            metrics = self.test()
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, loss, metrics)

        for callback in self.callbacks:
            callback.on_train_end(self.model)

    def train_epoch(self) -> torch.Tensor:
        total_loss = Mean(device=self.device)
        self.model.train()
        for i, (inputs, labels) in enumerate(self.train_data):
            for callback in self.callbacks:
                callback.on_batch_begin()

            model_input = inputs.to(self.device)
            ground_truth = labels.to(self.device)

            output = self.model(model_input)

            loss = self.loss_function(output, ground_truth)
            total_loss.update(loss)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            for callback in self.callbacks:
                callback.on_batch_end(i, total_loss.compute())

        return total_loss.compute()

    def test(self) -> dict[str : torch.Tensor]:
        self.model.eval()
        for tester in self.tester:
            tester.reset()

        for inputs, labels in self.test_data:
            model_input = inputs.to(self.device)
            ground_truth = labels.to(self.device)

            output = self.model(model_input)
            for tester in self.tester:
                tester.update(output, ground_truth)

        metrics = {}
        for tester in self.tester:
            metrics |= tester.compute()

        return metrics
