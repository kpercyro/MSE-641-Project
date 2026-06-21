import argparse
import csv
import os
import random
import re
import string
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pickle

def load_data(data_dir):
    """
    Load positive and negative reviews from the given directory.
    
    Args:
        data_dir (str): Path to directory containing pos.txt and neg.txt
        
    Returns:
        list: List of tuples (review_text, label)
    """
    # TODO: Implement this function
    
    labeled_data = []

    with open(data_dir, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)

        for row in reader:
            labeled_data.append((row[1], row[2]))
    
    #print(labeled_data[:5])
    return labeled_data

def shuffle_data(labeled_data):
    """
    Shuffle the labeled data.
    
    Args:
        labeled_data (list): List of (review_text, label) tuples
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Shuffled list of (review_text, label) tuples
    """
    # TODO: Implement this function
    # Shuffle the data while maintaining text-label pairs
    # Use the provided seed for reproducibility

    random.shuffle(labeled_data)

    # print(labeled_data[:5])
    return labeled_data
    
def tokenize(text):
    """
    Tokenize a text string.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    # TODO: Implement this function
    # Tokenize text into individual words
    # Remove specified special characters: !#$%&()*+/:,;.<=>@[\]^`{|}~\t\n
    tokens = []
    for special_char in "!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n":
        text = text.replace(special_char, "")
    tokenized_texts = text.split()
    return tokenized_texts

def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        tokenized_texts (list): List of tokenized texts
        labels (list): List of corresponding labels
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # TODO: Implement this function
    # Ensure the split is randomly sampled but reproducible
    # Check that ratios sum to 1.0
    # Maintain alignment between texts and their labels

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

    #--------------------------------------

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


# ---
def main():

    parser = argparse.ArgumentParser(description='Project Milestone')
    parser.add_argument('data_dir', type=str, help='Path to directory containing data.csv')
    args = parser.parse_args()

    data_dir = args.data_dir

    data_path = os.path.join(data_dir, 'data.csv')

    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Error: data.csv not found in {data_dir}")
        return
    
    # TODO: Implement the main workflow:
    # 1. Load data (combine positive and negative reviews with their labels)
    labeled_data = load_data(data_path)
    
    # 2. Shuffle the data
    shuffled_data = shuffle_data(labeled_data)
    
    # 3. Separate texts and labels
    texts = [item[1] for item in shuffled_data]
    labels = [item[0] for item in shuffled_data]
    
    # 4. Tokenize the texts
    tokenized_texts = [tokenize(text) for text in texts]
    
    # 5. Load stopwords
    # stopwords = load_stopwords(stopwords_path)
    
    # 6. Create version without stopwords
    # tokenized_texts_ns = [remove_stopwords(tokens, stopwords) for tokens in tokenized_texts]
    
    # 7. Split the data into train/val/test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_texts, labels)
    # train_texts_ns, val_texts_ns, test_texts_ns, _, _, _ = split_data(tokenized_texts_ns, labels)

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

    accuracy = evaluate_model(model, test_data_str, test_labels)
    print("accuracy", accuracy)

    stopword_name = "without_stopwords"

    model_filename = os.path.join("models.pkl")

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    feature_type = "unigrams"
            
    results.append(accuracy)

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()