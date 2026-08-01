from preprocessing import load_and_preprocess_data
from rnn import RNNModel
from gru import GRUModel
from lstm import LSTMModel
from bert import BertClassifier, prepare_bert_dataloaders, build_model_for
from dataloader import create_dataloaders
from train_transformer import train_transformer_model
import csv
import torch
import torch.nn as nn

DATA_PATH = "/workspaces/MSE-641-Project/data/data.csv"

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, test_loader, mlb = prepare_bert_dataloaders(
        DATA_PATH,
        tokenizer_name="distilbert-base-uncased",
        batch_size=8,
    )

    model = build_model_for(mlb, model_name="distilbert-base-uncased")

    lr = 2e-5  # standard fine-tuning LR for transformers, NOT 0.001

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    train_losses, val_losses = train_transformer_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
    )

    with open("/workspaces/MSE-641-Project/losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, train_loss, val_loss])

    print("Number of labels:", len(mlb.classes_))
    print("train_losses", train_losses)
    print("val_losses", val_losses)


if __name__ == "__main__":
    main()