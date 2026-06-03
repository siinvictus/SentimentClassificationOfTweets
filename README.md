# Sentiment Classification of Tweets

## Overview
This project builds a **binary sentiment classification pipeline** to predict whether a tweet expresses positive or negative sentiment. The work systematically compares **six different text preprocessing strategies** combined with **two classification pipelines**, evaluating each configuration by F1 score.

**Best result: F1 score = 0.846** using Multinomial Naive Bayes.

## Approach

### Text Preprocessing — 6 Configurations
Each solution notebook experiments with a different preprocessing format, varying combinations of:
- **Tokenization** strategies
- **Stop word removal**
- **Stemming / Lemmatization**
- **Count Vectorizer** vs **TF-IDF** feature extraction
- **N-gram ranges** (unigrams, bigrams)

### Classification Pipelines
Two pipelines tested across all preprocessing formats:
- **Multinomial Naive Bayes (MNB)** — probabilistic classifier well-suited to text frequency features
- **Stochastic Gradient Descent (SGD)** — linear classifier optimised with gradient descent, effective for high-dimensional sparse text data

### Evaluation
All models evaluated using **F1 score** — appropriate for binary classification where both precision and recall matter.

## How to Run

```bash
# Clone the repo
git clone https://github.com/siinvictus/SentimentClassificationOfTweets.git
cd SentimentClassificationOfTweets

# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter

# Then run any solution
jupyter notebook "Solution 6.ipynb"
```

## Tech Stack
Python, NLTK, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter

## Authors
Silva Bashllari & collaborators — Politecnico di Torino Course 

## License
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
