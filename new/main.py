from preprocessing import load_and_preprocess_data
from rnn import RNNModel
from train import train_model

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def main():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        tokenizer,
        mlb,
    ) = load_and_preprocess_data(
        "/workspaces/MSE-641-Project/data/data.csv"
    )

    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(mlb.classes_)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Number of Genres: {num_classes}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    print("Creating model...")
    model = RNNModel(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        output_dim=num_classes,
    )

    print("Model created")
    criterion = nn.BCEWithLogitsLoss()
    print("Criterion created")

    print("Model parameters:")

    for name, param in model.named_parameters():
        print(name, param.shape)

    print("Total params:",
        sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    print("Optimizer created")
    print("Starting training...")

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
    )

    print("Training Complete!")

    torch.save(
        model.state_dict(),
        "rnn_model.pth"
    )

    print("Model saved as rnn_model.pth")


if __name__ == "__main__":
    main()