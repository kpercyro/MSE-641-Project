import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as exc:
    raise ImportError(
        "transformers is required for BERT support. "
        "Install it with `pip install transformers`."
    ) from exc

from preprocessing import load_data, shuffle_data, split_data, encode_labels


class BertClassifier(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=None,
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return outputs.logits


def prepare_bert_dataloaders(
    data_path,
    tokenizer_name="distilbert-base-uncased",
    max_length=128,
    batch_size=8,
    seed=42,
):
    labeled_data = load_data(data_path)
    shuffled_data = shuffle_data(labeled_data, seed=seed)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    (
        train_texts,
        val_texts,
        test_texts,
        train_labels,
        val_labels,
        test_labels,
    ) = split_data(texts, labels)

    y_train, y_val, y_test, mlb = encode_labels(
        train_labels,
        val_labels,
        test_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def encode_texts(texts_list):
        encoded = tokenizer(
            texts_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded.get("token_type_ids"),
        )

    train_ids, train_mask, train_token_type_ids = encode_texts(train_texts)
    val_ids, val_mask, val_token_type_ids = encode_texts(val_texts)
    test_ids, test_mask, test_token_type_ids = encode_texts(test_texts)

    train_dataset = TensorDataset(
        train_ids,
        train_mask,
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        val_ids,
        val_mask,
        torch.FloatTensor(y_val),
    )
    test_dataset = TensorDataset(
        test_ids,
        test_mask,
        torch.FloatTensor(y_test),
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

    return train_loader, val_loader, test_loader, mlb
