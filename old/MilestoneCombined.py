import argparse
import csv
import os
import random
import re
import string
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pickle
from sklearn.metrics import f1_score

def load_data(data_dir):
    
    labeled_data = []

    with open(data_dir, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)

        for row in reader:
            labeled_data.append((row[1], row[2]))
    
    return labeled_data

def shuffle_data(labeled_data, seed=42):
   
    random.seed(seed)
    random.shuffle(labeled_data)
    return labeled_data

    
def tokenize(text):
    for special_char in "!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n":
        text = text.replace(special_char, "")
    tokenized_texts = text.split()
    return tokenized_texts

def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    
    train_count = int(train_ratio * len(labels))
    train_texts = list(tokenized_texts[:train_count])

    val_texts = []
    val_count = int(val_ratio * len(labels))
    for text in tokenized_texts[train_count:train_count + val_count]:
        val_texts.append(text)

    test_texts = []
    test_count = int(test_ratio * len(labels))
    for text in tokenized_texts[train_count + val_count:train_count + val_count + test_count]:
        test_texts.append(text)

    train_labels = []
    for label in labels[:train_count]:
        train_labels.append(label)

    val_labels = []
    for label in labels[train_count:train_count + val_count]:
        val_labels.append(label)

    test_labels = []
    for label in labels[train_count + val_count:train_count + val_count + test_count]:
        test_labels.append(label)

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    if use_unigrams and use_bigrams:
        ngram_range = (1,2)
    elif use_bigrams:
        ngram_range = (2,2)
    else:
        ngram_range = (1,1)
    
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    classifier = MultinomialNB()

    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    model.fit(train_data, train_labels)

    return model


def predict_top_k(model, texts, k=3):

    probabilities = model.predict_proba(texts)

    genre_classes = model.named_steps["classifier"].classes_

    top_k_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]

    top_k_predictions = []

    for movie_indices in top_k_indices:
        movie_predictions = genre_classes[movie_indices].tolist()
        top_k_predictions.append(movie_predictions)

    return top_k_predictions


def predict_top_k_with_probabilities(model, texts, k=3):

    probabilities = model.predict_proba(texts)

    genre_classes = model.named_steps["classifier"].classes_

    top_k_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]

    predictions_with_probs = []

    for row_number, movie_indices in enumerate(top_k_indices):
        movie_predictions = []

        for genre_index in movie_indices:
            genre = genre_classes[genre_index]
            probability = probabilities[row_number, genre_index]
            movie_predictions.append((genre, probability))

        predictions_with_probs.append(movie_predictions)

    return predictions_with_probs


def top_k_predictions_to_binary(top_k_predictions, genre_classes):
    
    binary_predictions = np.zeros((len(top_k_predictions), len(genre_classes)))

    genre_to_index = {}

    for index, genre in enumerate(genre_classes):
        genre_to_index[genre] = index

    for row_number, predicted_genres in enumerate(top_k_predictions):
        for genre in predicted_genres:
            genre_index = genre_to_index[genre]
            binary_predictions[row_number, genre_index] = 1

    return binary_predictions


def get_genre_classes(model):
    
    return model.named_steps["classifier"].classes_


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


def true_labels_to_binary(true_labels, genre_classes):
    
    binary_true_labels = np.zeros((len(true_labels), len(genre_classes)))

    genre_to_index = {}

    for index, genre in enumerate(genre_classes):
        genre_to_index[genre] = index

    for row_number, genre in enumerate(true_labels):     
        if genre in genre_to_index:
            genre_index = genre_to_index[genre]
            binary_true_labels[row_number, genre_index] = 1


    return binary_true_labels

def calculate_top_k_accuracy(top_k_predictions, true_labels):
    
    correct_predictions = 0

    for predicted_genres, true_genre in zip(top_k_predictions, true_labels):
        if true_genre in predicted_genres:
            correct_predictions += 1

    top_k_accuracy = correct_predictions / len(true_labels)

    return top_k_accuracy

def evaluate_top_k_predictions(top_k_predictions, true_labels, genre_classes):
    
    binary_predictions = top_k_predictions_to_binary(top_k_predictions, genre_classes)
    binary_true_labels = true_labels_to_binary(true_labels, genre_classes)

    top_k_accuracy = calculate_top_k_accuracy(top_k_predictions, true_labels)

    micro_f1 = f1_score(binary_true_labels, binary_predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(binary_true_labels, binary_predictions, average="macro", zero_division=0)

    results = {
        "top_k_accuracy": top_k_accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    return results


def main():

    parser = argparse.ArgumentParser(description='Project Milestone')
    parser.add_argument('data_dir', type=str, help='Path to directory containing data.csv')
    args = parser.parse_args()

    data_dir = args.data_dir

    data_path = os.path.join(data_dir, 'data.csv')

    if not os.path.exists(data_path):
        print(f"Error: data.csv not found in {data_dir}")
        return
    
    labeled_data = load_data(data_path)
    
    shuffled_data = shuffle_data(labeled_data)
    
    texts = [item[1] for item in shuffled_data]
    labels = [item[0] for item in shuffled_data]
    
    tokenized_texts = [tokenize(text) for text in texts]
    
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_texts, labels)
    
    print("Number of training samples:", len(train_labels))
    print("Number of validation samples:", len(val_labels))
    print("Number of test samples:", len(test_labels))

    print(train_texts[:5])
    print(train_labels[:5])

    results = []

    train_data = train_texts
    test_data = test_texts
    
    train_data_str = []
    for tokens in train_data:
        sentence = " ".join(tokens)
        train_data_str.append(sentence)
    
    test_data_str = []
    for tokens in test_data:
        sentence = " ".join(tokens)
        test_data_str.append(sentence)

    model = train_model(train_data_str, train_labels, use_bigrams=False, use_unigrams=True)
    
    stopword_name = "without_stopwords"

    model_filename = os.path.join("models.pkl")

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    feature_type = "unigrams"
    
    K = 3

    top_k_preds = predict_top_k(model, test_data_str, k=K)

    genre_classes = get_genre_classes(model)

    metrics = evaluate_top_k_predictions(top_k_preds, test_labels, genre_classes)

    print("Top-K predictions example:", top_k_preds[:5])
    print("Top-K accuracy:", metrics["top_k_accuracy"])
    print("Micro-F1:", metrics["micro_f1"])
    print("Macro-F1:", metrics["macro_f1"])

    results.append({
        "K": K,
        "Feature type": "unigrams",
        "Top-K accuracy": metrics["top_k_accuracy"],
        "Micro-F1": metrics["micro_f1"],
        "Macro-F1": metrics["macro_f1"]
    })

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()