import abc

import torch
from torcheval import metrics


def test(model, test_loader, device) -> metrics.MulticlassAccuracy:
    model.eval()
    accuracy = MulticlassAccuracy(device=device)
    confusion_matrix = MulticlassConfusionMatrix(4, device=device)
    for inputs, labels in test_loader:
        model_input = inputs.to(device)
        ground_truth = labels.to(device)

        output = model(model_input)
        accuracy.update(output.argmax(dim=1), ground_truth)
        confusion_matrix.update(output.argmax(dim=1), ground_truth)

    return accuracy, confusion_matrix


class Tester(abc.ABC):
    def update(
        self,
        _prediction: torch.Tensor,
        _ground_truth: torch.Tensor,
    ) -> tuple[str, torch.Tensor]: ...

    def compute(self) -> dict[str : torch.Tensor]: ...


class MulticlassAccuracy(Tester):
    def __init__(self, device: torch.device):
        self.device = device
        self.metric = metrics.MulticlassAccuracy(device=device)
        self.name = "MulticlassAccuracy"

    def update(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        self.metric.update(prediction.argmax(dim=1), ground_truth)

    def compute(self) -> dict[str : torch.Tensor]:
        return {self.name: self.metric.compute()}


class MulticlassConfusionMatrix(Tester):
    def __init__(self, classes: int, device: torch.device):
        self.device = device
        self.metric = metrics.MulticlassConfusionMatrix(classes, device=device)
        self.name = "MulticlassConfusionMatrix"

    def update(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        self.metric.update(prediction.argmax(dim=1), ground_truth)

    def compute(self) -> dict[str : torch.Tensor]:
        return {self.name: self.metric.compute()}
