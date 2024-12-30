import abc
import os

import torch
from torch import nn
from torch.utils.data import DataLoader


class Callback(abc.ABC):
    def on_train_begin(self, _train_data: DataLoader): ...

    def on_train_end(self, _model: nn.Module): ...

    def on_epoch_begin(
        self,
        _epoch: int,
    ): ...

    def on_epoch_end(
        self, _epoch: int, _train_loss: torch.Tensor, _metrics: dict[str:float]
    ): ...

    def on_batch_begin(self): ...

    def on_batch_end(self, _batch: int, _train_loss: torch.Tensor): ...


class ConsoleLogger(Callback):
    def __init__(self):
        self.epoch = 0

    def on_train_begin(self, train_data: DataLoader):
        print("--- Start training ---")
        print(f"Batches per epoch: {len(train_data)}")
        print()

    def on_train_end(self, _model):
        print()
        print("--- Training done ---")

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, _epoch, _train_loss, metrics: dict[str:float]):
        print()

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    def on_batch_end(self, batch: int, train_loss: torch.Tensor):
        loss_string = f"epoch {self.epoch}, batch {batch}: {train_loss:.4f}"
        print(loss_string, end="\r")


class ModelSaver(Callback):
    def __init__(self, path: str, params: list[str]):
        self.path = path
        self.params = [str(param) for param in params if param is not None]

    def on_train_end(self, model: nn.Module):
        file_name = f"model-{'_'.join(self.params)}.pth"
        out = os.path.join(self.path, file_name)
        print(f"Saving model to {out}")
        torch.save(model.state_dict(), out)
