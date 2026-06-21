import os
import pickle
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load tokenized data from a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [line.strip().split(',') for line in f]
    return data

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

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
    """
    Convert Top-K predictions into a binary multi-label matrix.

    This will be useful later for evaluation using micro-F1 and macro-F1.

    Example:
    genre_classes = ["action", "comedy", "drama", "thriller"]
    prediction = ["drama", "thriller"]

    binary row = [0, 0, 1, 1]
    """
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
    """
    Return the genre labels learned by the classifier.
    """
    return model.named_steps["classifier"].classes_


def save_model(model, file_path):
    """
    Save trained model to a pickle file.
    """
    folder = os.path.dirname(file_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path):
    """
    Load a saved model from a pickle file.
    """
    with open(file_path, "rb") as file:
        model = pickle.load(file)

    return model


#REMOVE ALL BELOW

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)

    correct_predictions = sum(predictions == test_labels)
    accuracy = correct_predictions / len(test_labels)

    return accuracy
    

def main():

    feature_options = [
        (True, False),  
        (False, True),  
        (True, True)    
    ]

    stopword_options = [True, False]
    
    train = load_data("data/train.csv")
    test = load_data("data/test.csv")

    train_ns = load_data("data/train_ns.csv")
    test_ns = load_data("data/test_ns.csv")

    train_labels = load_labels("data/train_labels.csv")
    test_labels = load_labels("data/test_labels.csv")

    results = []

    for use_unigrams, use_bigrams in feature_options:
        for remove_stopwords in stopword_options:
            if remove_stopwords:
                train_data = train_ns
                test_data = test_ns
            else:
                train_data = train
                test_data = test
            
            train_data_str = []
            for tokens in train_data:
                sentence = " ".join(tokens)
                train_data_str.append(sentence)
            
            test_data_str = []
            for tokens in test_data:
                sentence = " ".join(tokens)
                test_data_str.append(sentence)

            model = train_model(train_data_str, train_labels, use_bigrams, use_unigrams)

            accuracy = evaluate_model(model, test_data_str, test_labels)

            if use_unigrams and use_bigrams:
                feature_name = "unigrams_bigrams"
            elif use_unigrams:
                feature_name = "unigrams"
            else:
                feature_name = "bigrams"

            if remove_stopwords:
                stopword_name = "with_stopwords"
            else:
                stopword_name = "without_stopwords"

            model_filename = os.path.join("models", feature_name + "_" + stopword_name + ".pkl")

            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)

            if remove_stopwords:
                stopwords_value = "yes"
            else:
                stopwords_value = "no"

            if use_unigrams and use_bigrams:
                feature_type = "unigrams+bigrams"
            elif use_bigrams:
                feature_type = "bigrams"
            else:
                feature_type = "unigrams"
                   
            results.append({
                "Stopwords removed": stopwords_value,
                "text features": feature_type,
                "Accuracy (test set)": accuracy
            })

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()