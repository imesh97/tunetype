# Multinomial Naive Bayes Lyrics Classifier

This project implements a Multinomial Naive Bayes classifier to classify lyrics from the provided datasets.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Classifier

Run the classifier using the following command:
```
python src/naive_bayes_classifier.py
```

This will:
1. Load and preprocess the training data from `src/data/source/train.csv`
2. Load and preprocess the test data from `src/data/target/test.csv`
3. Train a Multinomial Naive Bayes classifier on the training data
4. Make predictions on the test data
5. Save the predictions to `src/data/target/predictions.csv`

## Model Details

The classifier uses:
- Text cleaning (removing special characters, converting to lowercase)
- CountVectorizer with English stop words removal and unigram/bigram features
- Multinomial Naive Bayes algorithm

This implementation is well-suited for text classification tasks like lyrics categorization. 