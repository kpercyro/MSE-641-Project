from preprocessing import load_and_preprocess_data
from rnn import RNNModel
from gru import GRUModel
from lstm import LSTMModel
from dataloader import create_dataloaders
from train import train_model
import torch
import torch.nn as nn

def main():

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
        "/workspaces/MSE-641-Project/data/data.csv"
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
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        output_dim=len(mlb.classes_),
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

    # print("Training shape:", X_train.shape)
    # print("Validation shape:", X_val.shape)
    # print("Test shape:", X_test.shape)
    # print("Label shape:", y_train.shape)
    # print("Vocabulary size:", len(vocab))
    # print("train_loader", train_loader)
    print("train_losses", train_losses)
    print("val_losses", val_losses)

if __name__ == "__main__":
    main()