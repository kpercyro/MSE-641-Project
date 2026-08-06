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

    if labels.ndim > 1:
        exact_match = np.mean(np.all(labels == preds, axis=1))
    else:
        exact_match = np.mean(labels == preds)

    metrics = {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "exact_match": float(exact_match),
    }

    return metrics, preds


def collect_probabilities(model, data_loader, device):
    """Collect model probabilities and labels for evaluation."""
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

    return probs, labels


def evaluate_model(model, data_loader, device, threshold=0.15, return_metrics=False):
    """Compute micro and macro F1 scores for multilabel classification."""
    probs, labels = collect_probabilities(model, data_loader, device)
    metrics, _ = evaluate_probabilities(probs, labels, threshold=threshold)

    if return_metrics:
        return metrics

    return metrics["micro_f1"], metrics["macro_f1"]


def evaluate_model_threshold_sweep(model, data_loader, device, thresholds=None):
    """Evaluate a model across a range of thresholds and return all results."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.5 + 1e-9, 0.05)

    probs, labels = collect_probabilities(model, data_loader, device)
    results = []

    for threshold in thresholds:
        threshold = float(threshold)
        metrics, _ = evaluate_probabilities(probs, labels, threshold=threshold)
        results.append(
            {
                "threshold": threshold,
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "exact_match": metrics["exact_match"],
            }
        )

    best_result = max(
        results,
        key=lambda item: (item["macro_f1"], item["micro_f1"], item["exact_match"]),
    )

    return results, best_result
