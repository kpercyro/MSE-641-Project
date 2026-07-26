from pathlib import Path

from preprocessing import load_and_preprocess_data
from bert import BERTModel
from dataloader import create_dataloaders
from train import train_model
from evaluate import evaluate_model
import torch
import torch.nn as nn


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {data_path}"
        )

    # preprocess
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        vocab,
        mlb,
    ) = load_and_preprocess_data(
        str(data_path)
    )

    # put preprocessed data in data loaders
    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=64,
    )

    # make model
    model = BERTModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=len(mlb.classes_),
        num_layers=2,
        num_heads=4,
        max_length=200,
    )

    # define training criterion
    criterion = nn.BCEWithLogitsLoss()

    # define device on which to train
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # define training approach
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
    )

    micro_f1, macro_f1 = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print("train_losses", train_losses)
    print("val_losses", val_losses)
    print(f"Test Micro F1: {micro_f1:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()