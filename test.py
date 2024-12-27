from torcheval.metrics import MulticlassAccuracy


def test(model, test_loader, device) -> MulticlassAccuracy:
    model.eval()
    accuracy = MulticlassAccuracy(device=device)
    for inputs, labels in test_loader:
        model_input = inputs.to(device)
        ground_truth = labels.to(device)

        output = model(model_input)
        accuracy.update(output.argmax(dim=1), ground_truth)

    return accuracy
