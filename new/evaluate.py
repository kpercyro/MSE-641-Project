import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_loss(labels, probabilities):
    """Compute binary cross-entropy loss for multilabel probabilities."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)

    return -np.mean(
        labels * np.log(probabilities)
        + (1 - labels) * np.log(1 - probabilities)
    )


def evaluate_probabilities(probabilities, labels, threshold=0.15):
    """Compute multilabel metrics from probability scores."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)

    preds = (probabilities >= threshold).astype(int)

    metrics = {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    return metrics, preds


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

    metrics, _ = evaluate_probabilities(probs, labels, threshold=threshold)

    return metrics["micro_f1"], metrics["macro_f1"]
