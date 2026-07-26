# preprocessing.py

import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_and_preprocess_data(
    csv_path,
    vocab_size=10000,
    max_length=200,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):

    df = pd.read_csv(csv_path)

    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")

    df["text"] = (
        df["overview"].astype(str)
        + " "
        + df["tagline"].astype(str)
    )

    df = df[df["genre"].notna()].copy()

    df["genre_list"] = df["genre"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()

    y = mlb.fit_transform(df["genre_list"])

    print("Genres:")
    print(mlb.classes_)
    print(f"Number of genres: {len(mlb.classes_)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"],
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - relative_val_size),
        random_state=random_state,
        shuffle=True,
    )

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    val_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        train_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_val_pad = pad_sequences(
        val_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_pad = pad_sequences(
        test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    return (
        X_train_pad,
        X_val_pad,
        X_test_pad,
        y_train,
        y_val,
        y_test,
        tokenizer,
        mlb,
    )


# if __name__ == "__main__":

#     (
#         X_train,
#         X_val,
#         X_test,
#         y_train,
#         y_val,
#         y_test,
#         tokenizer,
#         mlb,
#     ) = load_and_preprocess_data("/workspaces/MSE-641-Project/data/data.csv")

    #print("Training shape:", X_train.shape)
    #print("Validation shape:", X_val.shape)
    #print("Test shape:", X_test.shape)
    #print("Label shape:", y_train.shape)