# bertkate.py

from datetime import datetime

import torch

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from preprocessing import load_data_for_bert


def log(message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    message = f"[{timestamp}] {message}"

    print(message)

    with open(
        "katebert_output.txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(message + "\n")


class MovieGenreDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.float32,
            ),
        }


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0

    log("Starting epoch...")

    for batch_idx, batch in enumerate(train_loader):

        if batch_idx % 10 == 0:
            log(
                f"Batch {batch_idx}/{len(train_loader)}"
            )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    log("Epoch complete")

    return total_loss / len(train_loader)


def main():

    with open(
        "katebert_output.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("BERT Training Log\n\n")

    log("Loading data for BERT...")

    (
        train_texts,
        val_texts,
        test_texts,
        y_train,
        y_val,
        y_test,
        mlb,
    ) = load_data_for_bert("data/data.csv")

    log(f"Train size: {len(train_texts)}")
    log(f"Validation size: {len(val_texts)}")
    log(f"Test size: {len(test_texts)}")
    log(f"Number of genres: {len(mlb.classes_)}")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    train_dataset = MovieGenreDataset(
        train_texts,
        y_train,
        tokenizer,
    )

    val_dataset = MovieGenreDataset(
        val_texts,
        y_val,
        tokenizer,
    )

    test_dataset = MovieGenreDataset(
        test_texts,
        y_test,
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    log("Creating model...")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification",
    )

    log("Model created")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    log(f"Using device: {device}")

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    log("Optimizer created")
    log(
        f"Number of labels: "
        f"{model.config.num_labels}"
    )

    epochs = 1

    for epoch in range(epochs):

        log(
            f"Starting epoch "
            f"{epoch + 1}/{epochs}"
        )

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        log(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}"
        )

    log("Training complete")


if __name__ == "__main__":
    main()