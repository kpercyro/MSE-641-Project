# MSE 641 Project – Movie Genre Classification (Baseline)

## Overview
This project investigates the task of **predicting movie genres from plot descriptions** using natural language processing techniques. While movies can belong to multiple genres, the dataset provides only one observed genre per movie. To account for this, we implement a **Top‑K prediction approach**, where the model outputs a set of likely genres for each movie.

For the project milestone, we develop a **baseline model using Multinomial Naive Bayes** and evaluate its performance using multi-label metrics.

---

## Dataset
The dataset contains movie metadata with the following fields:

- **TITLE** – movie title  
- **GENRE** – one observed genre per movie  
- **DESCRIPTION** – plot synopsis  
- **DATE** – release year  

Although each movie has one labelled genre, it may realistically belong to multiple genres.

---

## Preprocessing Pipeline

The data processing pipeline includes:

1. **Loading data** from a CSV file  
2. **Shuffling** the dataset to ensure randomness  
3. **Tokenising text**:
   - Removing special characters  
   - Splitting into word tokens  
4. **Train/Validation/Test split** (80/10/10)  
5. **Reconstructing text strings** for model input  

---

## Baseline Model

We implement a **Multinomial Naive Bayes classifier** using a scikit-learn pipeline:

- **CountVectorizer**
  - Converts text into bag-of-words features  
  - Supports unigram and bigram configurations  
- **MultinomialNB**
  - Predicts a genre based on input text  

---

## Top‑K Prediction Strategy

To align with the **multi-label nature of movie genres**, we extend the baseline model:

- The model outputs **probabilities for all genres**
- We select the **Top‑K highest-probability genres** (K = 3)
- Each movie is assigned a **set of predicted genres**

## AI Declaration
We used Microsoft Copilot for debugging and to generate the README for this project. 