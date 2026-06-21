import argparse
import csv
import os
import random
import re
import string


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


# def load_stopwords(stopwords_path):
#     """
#     Load stopwords from the provided file.
    
#     Args:
#         stopwords_path (str): Path to the stopwords file
        
#     Returns:
#         set: Set of stopwords
#     """
#     # TODO: Implement this function
#     # Load and return stopwords from the provided file path
#     # Handle file not found errors gracefully
#     with open(stopwords_path, "r") as f:
#         stopwords = [line.strip() for line in f.readlines()]
#     #print(stopwords[:5])
#     return stopwords


# def remove_stopwords(tokens, stopwords):
#     """
#     Remove stopwords from a list of tokens.
    
#     Args:
#         tokens (list): List of tokens
#         stopwords (set): Set of stopwords
        
#     Returns:
#         list: List of tokens with stopwords removed
#     """
#     # TODO: Implement this function
#     # Remove any token that appears in the stopwords set
#     tokens_no_stopwords = []
#     for token in tokens:
#         if token.lower() not in stopwords:
#             tokens_no_stopwords.append(token)
#     return tokens_no_stopwords


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

def write_to_csv(tokenized_texts, output_file):
    """
    Write tokenized texts to a CSV file.
    
    Args:
        tokenized_texts (list): List of token lists
        output_file (str): Path to output file
    """
    # TODO: Implement this function
    # Write each list of tokens as a comma-separated string
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for tokens in tokenized_texts:
            writer.writerow(tokens)


def write_labels_to_csv(labels, output_file):
    """
    Write labels to a CSV file.
    
    Args:
        labels (list): List of labels
        output_file (str): Path to output file
    """
    # TODO: Implement this function
    # Write each label on a separate line
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])


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
    
    # # 8. Write tokenized texts and labels to CSV files
    # write_to_csv(tokenized_texts, os.path.join(data_dir, 'out.csv'))
    # write_to_csv(train_texts, os.path.join(data_dir, 'train.csv'))
    # write_to_csv(val_texts, os.path.join(data_dir, 'val.csv'))
    # write_to_csv(test_texts, os.path.join(data_dir, 'test.csv'))
    
    # write_to_csv(tokenized_texts_ns, os.path.join(data_dir, 'out_ns.csv'))
    # write_to_csv(train_texts_ns, os.path.join(data_dir, 'train_ns.csv'))
    # write_to_csv(val_texts_ns, os.path.join(data_dir, 'val_ns.csv'))
    # write_to_csv(test_texts_ns, os.path.join(data_dir, 'test_ns.csv'))
    
    # write_labels_to_csv(labels, os.path.join(data_dir, 'out_labels.csv'))
    # write_labels_to_csv(train_labels, os.path.join(data_dir, 'train_labels.csv'))
    # write_labels_to_csv(val_labels, os.path.join(data_dir, 'val_labels.csv'))
    # write_labels_to_csv(test_labels, os.path.join(data_dir, 'test_labels.csv'))
    
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()