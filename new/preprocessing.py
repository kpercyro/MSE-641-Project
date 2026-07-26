# preprocessing.py

import ast
import csv
import random
from collections import Counter

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer


def load_data(data_path):

    labeled_data = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            overview = row["overview"]
            genre_string = row["genre"]

            if overview is None or genre_string is None:
                continue

            overview = overview.strip()
            genre_string = genre_string.strip()

            if overview == "" or genre_string == "":
                continue

            try:
                genre_list = ast.literal_eval(genre_string)
            except ValueError:
                continue
            except SyntaxError:
                continue

            if len(genre_list) == 0:
                continue

            labeled_data.append((overview, genre_list))

    return labeled_data


def shuffle_data(labeled_data, seed=42):

    random.seed(seed)
    random.shuffle(labeled_data)

    return labeled_data


def tokenize(text):

    for special_char in "!#$%&()*+/:,;.<=>?@[\\]^`{|}~\t\n":
        text = text.replace(special_char, "")

    tokenized_text = text.lower().split()

    return tokenized_text


def split_data(
    tokenized_texts,
    labels,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):

    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1.0"
        )

    train_count = int(train_ratio * len(labels))
    val_count = int(val_ratio * len(labels))

    train_texts = tokenized_texts[:train_count]
    val_texts = tokenized_texts[
        train_count : train_count + val_count
    ]
    test_texts = tokenized_texts[
        train_count + val_count :
    ]

    train_labels = labels[:train_count]
    val_labels = labels[
        train_count : train_count + val_count
    ]
    test_labels = labels[
        train_count + val_count :
    ]

    return (
        train_texts,
        val_texts,
        test_texts,
        train_labels,
        val_labels,
        test_labels,
    )


def encode_labels(
    train_labels,
    val_labels,
    test_labels,
):

    mlb = MultiLabelBinarizer()

    train_labels_binary = mlb.fit_transform(train_labels)
    val_labels_binary = mlb.transform(val_labels)
    test_labels_binary = mlb.transform(test_labels)

    return (
        train_labels_binary,
        val_labels_binary,
        test_labels_binary,
        mlb,
    )


def build_vocab(train_texts):

    counter = Counter()

    for tokens in train_texts:
        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<OOV>": 1,
    }

    for idx, (word, _) in enumerate(
        counter.most_common(),
        start=2,
    ):
        vocab[word] = idx

    return vocab


def convert_to_ids(tokenized_texts, vocab):

    sequences = []

    for tokens in tokenized_texts:

        sequence = [
            vocab.get(token, vocab["<OOV>"])
            for token in tokens
        ]

        sequences.append(sequence)

    return sequences


def pad_sequences(
    sequences,
    max_length=200,
):

    padded = np.zeros(
        (len(sequences), max_length),
        dtype=np.int64,
    )

    for i, sequence in enumerate(sequences):

        sequence = sequence[:max_length]

        padded[i, : len(sequence)] = sequence

    return padded


def load_and_preprocess_data(
    data_path,
    max_length=200,
):

    labeled_data = load_data(data_path)

    shuffled_data = shuffle_data(labeled_data)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    tokenized_texts = [
        tokenize(text)
        for text in texts
    ]

    (
        train_texts,
        val_texts,
        test_texts,
        train_labels,
        val_labels,
        test_labels,
    ) = split_data(
        tokenized_texts,
        labels,
    )

    (
        y_train,
        y_val,
        y_test,
        mlb,
    ) = encode_labels(
        train_labels,
        val_labels,
        test_labels,
    )

    vocab = build_vocab(train_texts)

    train_sequences = convert_to_ids(
        train_texts,
        vocab,
    )

    val_sequences = convert_to_ids(
        val_texts,
        vocab,
    )

    test_sequences = convert_to_ids(
        test_texts,
        vocab,
    )

    X_train = pad_sequences(
        train_sequences,
        max_length=max_length,
    )

    X_val = pad_sequences(
        val_sequences,
        max_length=max_length,
    )

    X_test = pad_sequences(
        test_sequences,
        max_length=max_length,
    )

    print("Genres:")
    print(mlb.classes_)
    print(f"Number of genres: {len(mlb.classes_)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        vocab,
        mlb,
    )


if __name__ == "__main__":

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

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print("Label shape:", y_train.shape)
    print("Vocabulary size:", len(vocab))