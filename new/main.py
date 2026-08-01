from pathlib import Path

from preprocessing import load_and_preprocess_data
from rnn import RNNModel
from gru import GRUModel
from lstm import LSTMModel
from dataloader import create_dataloaders
from train import train_model
from evaluate import evaluate_model
import torch
import torch.nn as nn

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "data_500.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {data_path}"
        )

    MODEL_TYPE = "lstm"  # change this to "rnn", "gru", or "lstm"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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
        data_path,
    )

    # Print first 2 samples of X_train and y_train for verification
    # print("First 2 samples of X_train:", X_train[:2])
    # print("First 2 samples of y_train:", y_train[:2])

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=8,
    )

    # make model
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        output_dim=len(mlb.classes_),
        num_layers=2,
    )

    print("done making model")

    # train model

    lr = 0.01

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

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

    # print("Training shape:", X_train.shape)
    # print("Validation shape:", X_val.shape)
    # print("Test shape:", X_test.shape)
    # print("Label shape:", y_train.shape)
    # print("Vocabulary size:", len(vocab))
    # print("train_loader", train_loader)
    print("train_losses", train_losses)
    print("val_losses", val_losses)
    print(f"Test Micro F1: {micro_f1:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    # Check whether the model is predicting all zeros on the test set
    all_zero_predictions = True
    total_positive_predictions = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.25).int()
            total_positive_predictions += int(preds.sum().item())

    all_zero_predictions = total_positive_predictions == 0
    print("All-zero predictions on test set:", all_zero_predictions)
    print("Positive predictions on test set:", total_positive_predictions)


if __name__ == "__main__":
    main()