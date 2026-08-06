# MSE 641 Project - Movie Genre Classification

## Overview
This project explores whether plot descriptions can be used to predict movie genres. Because movies often belong to more than one genre, the task is treated as a multilabel classification problem rather than a single-label problem.

The repository contains a full workflow for preprocessing text data, training several NLP models, and evaluating multilabel predictions using metrics such as micro-F1, macro-F1, and exact match.

---

## Project Goal
The main objective is to build and compare models that predict a set of relevant genres for each movie from its synopsis.

The project includes:
- a baseline machine-learning approach,
- neural sequence models such as RNN, GRU, and LSTM-style architectures,
- and a transformer-based BERT experiment in the notebook.

---

## Dataset
The project uses a movie dataset stored in the data folder.

Primary files:
- data/data.csv
- data/data_500.csv

The expected columns include movie text and genre information, with the code reading:
- overview / synopsis text
- genre labels

The data is split into train, validation, and test sets before model training.

---

## Repository Structure
- Baseline/ - baseline modeling experiments
- data/ - dataset files and preprocessing utilities
- new/ - main deep-learning implementation
  - preprocessing.py - text cleaning, tokenization, label encoding, and padding
  - dataloader.py - PyTorch DataLoader creation
  - rnn.py - recurrent neural network baseline
  - gru.py - GRU-based model
  - lstm.py - LSTM-based model implementation
  - train.py - training loop and loss plotting
  - evaluate.py - multilabel evaluation and threshold sweep logic
  - main.py - end-to-end training and evaluation entry point
  - mse-641-bert-model.ipynb - transformer-based BERT experiment
- old/ - baseline model and outputs

---

## Environment Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements include:
- numpy
- pandas
- scikit-learn
- torch
- transformers
- matplotlib

---

## Preprocessing Pipeline
The preprocessing workflow includes:
1. loading movie descriptions and genre labels from CSV files,
2. cleaning and tokenizing the text,
3. shuffling data for randomness,
4. splitting into train/validation/test subsets,
5. converting genre labels into binary multilabel vectors,
6. building a vocabulary and padding sequences for neural models.

---

## Modeling Approaches

### 1. Baseline Models
The baseline experiments use classical machine-learning methods and are stored in the Baseline folder.

### 2. Neural Sequence Models
The new implementation trains recurrent neural networks on tokenized text sequences.

Supported architectures include:
- RNN
- GRU
- LSTM

These models use embedding layers and sigmoid outputs for multilabel prediction.

### 3. Transformer-Based Model
The notebook in the new folder explores a BERT-based multilabel classifier using Hugging Face Transformers.
Because training and storing transformer models can require significant computational resources, the BERT experiment was developed and executed in Kaggle using GPU acceleration. The notebook includes training, validation, threshold optimization, and final test evaluation for the multilabel genre classification task.

The project on Kaggle can be accessed here: https://www.kaggle.com/code/etakpr/mse-641-bert-model
---

## Training and Evaluation
Training is performed with binary cross-entropy loss for multilabel output, and evaluation uses metrics such as:
- micro-F1
- macro-F1
- exact match

The evaluation scripts also perform a threshold sweep to select the best decision threshold for converting probabilities into genre predictions.

---

## How to Run
From the project root, run:

```bash
python new/main.py
```

This script will:
- load the dataset,
- preprocess the text,
- train a neural model,
- evaluate it on the test set,
- and print the best threshold and metric values.

To explore the transformer approach, open and run the notebook:
- new/mse-641-bert-model.ipynb

---

## Outputs
The project generates:
- model training logs,
- loss plots such as loss_plot.png and loss_plot_2_layer.png,
- evaluation metrics for different thresholds,
- and saved artifacts from the BERT experiment when run in the notebook.

---

## Artificial Intelligence Declaration
Generative AI tools (Microsoft Copilot/ChatGPT/Claude) were used to assist with code debugging, troubleshooting implementation issues, and improving the clarity of documentation such as the project README. All design decisions, model development, experimentation, evaluation, and final project content were reviewed and verified by the authors. The authors take full responsibility for the accuracy and integrity of the submitted work.
