# Sentiment Analysis of Gojek App Reviews

## Overview

This project analyzes sentiment in Indonesian-language user reviews of the Gojek mobile application.

The analysis focuses on reviews from Gojek app version 4.8.x and applies Natural Language Processing (NLP) techniques for text preprocessing, sentiment labeling, exploratory analysis, and machine learning classification.

The sentiment labels are generated automatically using VADER with an additional custom Indonesian sentiment lexicon.

## Objectives

- Analyze sentiment patterns in Indonesian Gojek app reviews.
- Classify reviews into Positive, Neutral, and Negative sentiment categories.
- Explore frequently occurring words across different sentiment categories.
- Transform text into numerical features using TF-IDF.
- Build a Random Forest classification model using the generated sentiment labels.
- Evaluate model performance on a held-out test set.

## Dataset

The dataset used in this project is:

**Gojek App Reviews Bahasa Indonesia**

Source:

Kaggle: https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia

For this analysis, only reviews from Gojek application version **4.8.x** are used.

The dataset contains review-related variables including:

- `userName`: Username or alias of the reviewer.
- `content`: Text of the Gojek user review.
- `score`: Star rating provided by the user.
- `appVersion`: Version of the Gojek application.

The analysis focuses primarily on the review text and sentiment.

## Methodology

The project follows these main steps:

1. Load the review dataset.
2. Filter reviews from Gojek application version 4.8.x.
3. Select the relevant variables.
4. Remove missing and duplicate review text.
5. Tokenize the review text.
6. Remove Indonesian, English, and custom informal stopwords.
7. Apply Indonesian stemming using Sastrawi.
8. Generate sentiment labels using VADER and a custom Indonesian sentiment lexicon.
9. Explore sentiment distribution and word frequency.
10. Split the text data into training and test sets using stratification.
11. Convert text into numerical features using TF-IDF.
12. Train a Random Forest classifier.
13. Apply SMOTE inside the cross-validation pipeline to address class imbalance.
14. Tune Random Forest hyperparameters using RandomizedSearchCV.
15. Evaluate the final model on the held-out test set.

## Sentiment Labeling

Sentiment labels are generated automatically using VADER.

Because standard VADER is primarily designed for English text, a custom Indonesian sentiment lexicon is added to the analyzer.

Examples of Indonesian terms included in the custom lexicon include:

- `kecewa`
- `buruk`
- `jelek`
- `lelet`
- `gagal`
- `bagus`
- `baik`
- `mantap`
- `puas`
- `lambat`
- `susah`

The resulting sentiment categories are:

- **Positif**
- **Netral**
- **Negatif**

## Exploratory Analysis

The project includes exploratory visualizations such as:

- Sentiment distribution
- Sentiment bar chart
- Word clouds for Neutral, Positive, and Negative reviews
- Word-frequency visualizations for each sentiment category
- Sentiment distribution funnel chart

## Machine Learning

The text classification workflow uses:

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is used to transform the processed review text into numerical feature vectors.

The TF-IDF vectorizer is fitted only on the training text and then applied to the held-out test text.

### Random Forest

A Random Forest classifier is trained using the TF-IDF features.

Hyperparameters are tuned using `RandomizedSearchCV` with 5-fold cross-validation.

### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is applied inside the imbalanced-learn pipeline during cross-validation to address class imbalance without generating synthetic samples from the held-out test set.

## Evaluation

The final model is evaluated on the held-out test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

The notebook contains the resulting evaluation metrics.

## Limitation

The sentiment labels in this project are **automatically generated pseudo-labels** using VADER combined with a custom Indonesian lexicon.

They are **not human-annotated sentiment labels**.

Therefore, the machine learning evaluation metrics should be interpreted as the model's agreement with the automatically generated labels, rather than as accuracy against human-annotated sentiment.
