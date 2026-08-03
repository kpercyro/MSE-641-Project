# preprocessing.py

import ast
import csv
import random
from collections import Counter

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

try:
    from nltk.corpus import stopwords as nltk_stopwords

    STOPWORDS = set(nltk_stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "from", "in", "into", "of", "on", "to", "with", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "our", "their", "as", "so", "such", "not", "no", "nor", "too", "very", "can", "could", "would", "should", "will", "just", "do", "does", "did", "have", "has", "had", "may", "might", "must", "who", "whom", "whose", "what", "which", "where", "when", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "only", "own", "same", "than", "too", "very", "s", "t", "don", "should", "now"
    }


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


def tokenize(text, remove_stopwords=True):

    for special_char in "!#$%&()*+/:,;.<=>?@[\\]^`{|}~\t\n":
        text = text.replace(special_char, "")

    tokenized_text = text.lower().split()

    if remove_stopwords:
        tokenized_text = [
            token for token in tokenized_text if token not in STOPWORDS
        ]

    return tokenized_text


def split_data(
    tokenized_texts,
    labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
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
    remove_stopwords=True,
):

    labeled_data = load_data(data_path)

    shuffled_data = shuffle_data(labeled_data)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    tokenized_texts = [
        tokenize(text, remove_stopwords=remove_stopwords)
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

    # TESTING
    print("first tokenized text:", tokenized_texts[0])

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