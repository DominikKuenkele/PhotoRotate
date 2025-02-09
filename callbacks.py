import abc
import os

import torch
from torch import nn
from torch.utils.data import DataLoader


class Callback(abc.ABC):
    def on_train_begin(self, _model: nn.Module, _train_data: DataLoader): ...

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

    def on_train_begin(self, _model, train_data: DataLoader):
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


class FileLogger(Callback):
    def __init__(self, path: str, params: list[str]):
        file_name = f"training_{'_'.join([str(param) for param in params if param is not None])}"
        self.file = os.path.join(path, file_name)
        
        suffix = 0
        while os.path.exists(os.path.join(path, f"{file_name}.log")):
            file_name = file_name.removesuffix(f"_{str(suffix)}")
            suffix+=1
            file_name += f"_{str(suffix)}"

        self.file = os.path.join(path, f"{file_name}.log")
        with open(self.file, "w", encoding="utf-8"):
            pass

        self.epoch = 0

    def on_train_begin(self, model: nn.Module, _train_data):
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(f"{model}\n")

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, _epoch, train_loss, metrics: dict[str:float]):
        loss_string = f"epoch {self.epoch}: {train_loss:.4f}"
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(f"{loss_string}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")


class ModelSaver(Callback):
    def __init__(self, path: str, params: list[str], as_state_dict: bool):
        self.as_state_dict = as_state_dict
        
        file_name = f"model_{'_'.join([str(param) for param in params if param is not None])}"

        suffix = 0
        while os.path.exists(os.path.join(path, f"{file_name}.pth")):
            file_name = file_name.removesuffix(f"_{str(suffix)}")
            suffix+=1
            file_name += f"_{str(suffix)}"

        self.file = os.path.join(path, f"{file_name}.pth")


    def on_train_end(self, model: nn.Module):
        print(f"Saving model to {self.file}")
        if self.as_state_dict:
            torch.save(model.state_dict(), self.file)
        else:
            torch.save(model, self.file)
