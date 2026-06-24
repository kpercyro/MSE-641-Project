import argparse
import ast
import csv
import os
import random
import pandas as pd
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


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
    for special_char in "!#$%&()*+/:,;.<= >@[\]^`{|}~\t\n":
        text = text.replace(special_char, "")
    tokenized_texts = text.split()
    return tokenized_texts


def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    train_count = int(train_ratio * len(labels))
    val_count = int(val_ratio * len(labels))

    train_texts = tokenized_texts[:train_count]
    val_texts = tokenized_texts[train_count:train_count + val_count]
    test_texts = tokenized_texts[train_count + val_count:]

    train_labels = labels[:train_count]
    val_labels = labels[train_count:train_count + val_count]
    test_labels = labels[train_count + val_count:]

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def convert_tokens_to_strings(tokenized_texts):
    text_strings = []

    for tokens in tokenized_texts:
        sentence = " ".join(tokens)
        text_strings.append(sentence)

    return text_strings


def encode_labels(train_labels, val_labels, test_labels):

    mlb = MultiLabelBinarizer()

    train_labels_binary = mlb.fit_transform(train_labels)
    val_labels_binary = mlb.transform(val_labels)
    test_labels_binary = mlb.transform(test_labels)

    return train_labels_binary, val_labels_binary, test_labels_binary, mlb


def train_model(train_data, train_labels_binary, use_bigrams=False, use_unigrams=True):
    if use_unigrams and use_bigrams:
        ngram_range = (1, 2)
    elif use_bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    classifier = OneVsRestClassifier(MultinomialNB())

    model = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])

    model.fit(train_data, train_labels_binary)

    return model


def evaluate_model(model, test_data, test_labels_binary):

    probs = model.predict_proba(test_data)
    predictions = (probs > 0.2).astype(int)

    exact_match_accuracy = accuracy_score(test_labels_binary, predictions)
    micro_f1 = f1_score(test_labels_binary, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(test_labels_binary, predictions, average="macro", zero_division=0)

    results = {
        "exact_match_accuracy": exact_match_accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    return results, predictions


def save_model(model, file_path):
    folder = os.path.dirname(file_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)

    return model


def main():
    parser = argparse.ArgumentParser(description="Project Milestone")
    parser.add_argument("data_dir", type=str, help="Path to directory containing data.csv")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_path = os.path.join(data_dir, "data.csv")

    if not os.path.exists(data_path):
        print(f"Error: data.csv not found in {data_dir}")
        return

    labeled_data = load_data(data_path)

    shuffled_data = shuffle_data(labeled_data)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    tokenized_texts = [tokenize(text) for text in texts]

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        tokenized_texts,
        labels
    )

    print("Number of total samples:", len(labels))
    print("Number of training samples:", len(train_labels))
    print("Number of validation samples:", len(val_labels))
    print("Number of test samples:", len(test_labels))

    print("Example training text:", train_texts[0])
    print("Example training labels:", train_labels[0])

    train_data_str = convert_tokens_to_strings(train_texts)
    val_data_str = convert_tokens_to_strings(val_texts)
    test_data_str = convert_tokens_to_strings(test_texts)

    train_labels_binary, val_labels_binary, test_labels_binary, mlb = encode_labels(
        train_labels,
        val_labels,
        test_labels
    )

    model = train_model(
        train_data_str,
        train_labels_binary,
        use_bigrams=False,
        use_unigrams=True
    )

    metrics, predictions = evaluate_model(model, test_data_str, test_labels_binary)

    predicted_label_sets = mlb.inverse_transform(predictions)

    print("Example predicted labels:", predicted_label_sets[:5])
    print("Example true labels:", test_labels[:5])

    print("Exact-match accuracy:", metrics["exact_match_accuracy"])
    print("Micro-F1:", metrics["micro_f1"])
    print("Macro-F1:", metrics["macro_f1"])

    save_model(model, "models.pkl")

    results = [{
        "Feature type": "unigrams",
        "Exact-match accuracy": metrics["exact_match_accuracy"],
        "Micro-F1": metrics["micro_f1"],
        "Macro-F1": metrics["macro_f1"]
    }]

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

    print("Model training and evaluation completed successfully.")


if __name__ == "__main__":
    main()