import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    batch_size=64,
):
    """
    Convert preprocessed numpy arrays into
    PyTorch DataLoaders.
    """

    # Inputs must be LongTensor for Embedding layer
    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(X_val)
    X_test = torch.LongTensor(X_test)

    # Labels must be FloatTensor for BCEWithLogitsLoss
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    val_dataset = TensorDataset(
        X_val,
        y_val,
    )

    test_dataset = TensorDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )