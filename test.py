from torcheval.metrics import BinaryAccuracy

def test(model, test_loader, device) -> BinaryAccuracy:
    model.eval()
    accuracy = BinaryAccuracy()
    for inputs, labels in test_loader:
        model_input = inputs.to(device)
        ground_truth = labels.to(device)

        output = model(model_input).squeeze()
        accuracy.update(output, ground_truth.float())

    return accuracy