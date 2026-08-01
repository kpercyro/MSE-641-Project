import torch
from sklearn.metrics import f1_score


def evaluate_model(model, data_loader, device, threshold=0.15):
    """Compute micro and macro F1 scores for multilabel classification."""
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    preds = (probs >= threshold).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return micro_f1, macro_f1
