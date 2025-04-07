import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re

# Function to clean lyrics text
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Load the training data
print("Loading and preprocessing training data...")
train_data = pd.read_csv('src/data/source/train.csv')
train_data['lyric'] = train_data['lyric'].apply(clean_text)

# Load the test data
print("Loading and preprocessing test data...")
test_data = pd.read_csv('src/data/target/test.csv')
test_data['lyric'] = test_data['lyric'].apply(clean_text)

# Create a pipeline with CountVectorizer and MultinomialNB
print("Creating and training the Multinomial Naive Bayes model...")
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(train_data['lyric'], train_data['class'])

# Predict on the test data
print("Making predictions on test data...")
predictions = pipeline.predict(test_data['lyric'])

# Save results to CSV in the required format
print("Saving predictions to CSV...")
results = pd.DataFrame({
    'id': test_data['id'],
    'class': predictions
})
results.to_csv('src/results/mnb_predictions.csv', index=False)

print("Classification complete! Results saved to src/results/mnb_predictions.csv")

# If we had labeled test data, we could evaluate performance:
# accuracy = accuracy_score(test_data['class'], predictions)
# print(f"Accuracy: {accuracy:.4f}")
# print(classification_report(test_data['class'], predictions)) 