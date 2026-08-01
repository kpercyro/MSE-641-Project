import argparse
import ast
import csv
import os
import pickle
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_PREPROCESSING_PATH = PROJECT_ROOT / "new"
if str(NEW_PREPROCESSING_PATH) not in sys.path:
    sys.path.insert(0, str(NEW_PREPROCESSING_PATH))

from preprocessing import (
    load_data,
    shuffle_data,
    tokenize,
    split_data,
    encode_labels,
)
from evaluate import compute_loss, evaluate_probabilities

def convert_tokens_to_strings(tokenized_texts):
    return [" ".join(tokens) for tokens in tokenized_texts]

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
        ("classifier", classifier),
    ])

    model.fit(train_data, train_labels_binary)

    return model


def select_threshold(model, train_data, val_data, train_labels_binary, val_labels_binary):
    train_probabilities = model.predict_proba(train_data)
    val_probabilities = model.predict_proba(val_data)

    train_loss = compute_loss(
        train_labels_binary,
        train_probabilities,
    )

    val_loss = compute_loss(
        val_labels_binary,
        val_probabilities,
    )

    best_threshold = 0.5
    best_score = -1.0

    thresholds = [
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90
    ]

    print("\nValidation Results")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print("-" * 90)
    print(
        f"{'Threshold':<12}"
        f"{'Micro-F1':<12}"
        f"{'Macro-F1':<12}"
    )

    for threshold in thresholds:
        metrics, val_predictions = evaluate_probabilities(
            val_probabilities,
            val_labels_binary,
            threshold=threshold,
        )

        micro_f1 = metrics["micro_f1"]
        macro_f1 = metrics["macro_f1"]

        print(
            f"{threshold:<12.2f}"
            f"{micro_f1:<12.4f}"
            f"{macro_f1:<12.4f}"
        )

        if macro_f1 > best_score:
            best_score = macro_f1
            best_threshold = threshold

    print("-" * 90)
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation Macro-F1: {best_score:.4f}\n")

    return best_threshold


def evaluate_model(model, test_data, test_labels_binary, threshold):
    probs = model.predict_proba(test_data)
    metrics, predictions = evaluate_probabilities(
        probs,
        test_labels_binary,
        threshold=threshold,
    )

    exact_match_accuracy = accuracy_score(test_labels_binary, predictions)

    results = {
        "exact_match_accuracy": exact_match_accuracy,
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "threshold": threshold,
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
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for saving the trained model and metrics")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / "data.csv"

    if not data_path.exists():
        print(f"Error: data.csv not found in {data_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "outputs/new"
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_data = load_data(data_path)
    shuffled_data = shuffle_data(labeled_data)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    tokenized_texts = [tokenize(text) for text in texts]

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        tokenized_texts,
        labels,
    )

    print("Number of total samples:", len(labels))
    print("Number of training samples:", len(train_labels))
    print("Number of validation samples:", len(val_labels))
    print("Number of test samples:", len(test_labels))

    train_data_str = convert_tokens_to_strings(train_texts)
    val_data_str = convert_tokens_to_strings(val_texts)
    test_data_str = convert_tokens_to_strings(test_texts)

    train_labels_binary, val_labels_binary, test_labels_binary, mlb = encode_labels(
        train_labels,
        val_labels,
        test_labels,
    )

    model = train_model(train_data_str, train_labels_binary, use_bigrams=False, use_unigrams=True)
    threshold = select_threshold(model, train_data_str, val_data_str, train_labels_binary, val_labels_binary)
    metrics, predictions = evaluate_model(model, test_data_str, test_labels_binary, threshold)

    predicted_label_sets = mlb.inverse_transform(predictions)

    print("Selected threshold:", threshold)
    print("Example predicted labels:", predicted_label_sets[:5])
    print("Example true labels:", test_labels[:5])
    print("Exact-match accuracy:", metrics["exact_match_accuracy"])
    print("Micro-F1:", metrics["micro_f1"])
    print("Macro-F1:", metrics["macro_f1"])

    save_model(model, output_dir / "model.pkl")

    results = [{
        "Feature type": "unigrams",
        "Threshold": metrics["threshold"],
        "Exact-match accuracy": metrics["exact_match_accuracy"],
        "Micro-F1": metrics["micro_f1"],
        "Macro-F1": metrics["macro_f1"],
    }]

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)

    print("Model training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
