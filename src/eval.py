"""
TuneType - Model Evaluation
Retrains models with proper training data and compares model predictions against ground truth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import os
import pickle
import sys

# Add text cleaning functions
import re
import string

def clean_text(text):
    """Basic text cleaning function"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Define paths
TRAIN_DATA_PATH = "src/data/source/train.csv"  # Training data
TEST_DATA_PATH = "src/data/target/test.csv"    # Test data with lyrics
GROUND_TRUTH_PATH = "src/data/target/sample.csv"  # Test data with class labels
OUTPUT_DIR = "plots"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load ground truth (for evaluation)
ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
print(f"Ground truth class distribution: {ground_truth['class'].value_counts().to_dict()}")

# Load test data (for prediction)
test_data = pd.read_csv(TEST_DATA_PATH)
print(f"Test data loaded with {len(test_data)} samples")

# Load training data
train_data = pd.read_csv(TRAIN_DATA_PATH)
print(f"Training data loaded with {len(train_data)} samples")
print(f"Training data class distribution: {train_data['class'].value_counts().to_dict()}")

# Clean the text data
print("Cleaning text data...")
train_data['lyric'] = train_data['lyric'].apply(clean_text)
test_data['lyric'] = test_data['lyric'].apply(clean_text)

# Train Multinomial Naive Bayes
print("Training Multinomial Naive Bayes model...")
mnb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])
mnb_pipeline.fit(train_data['lyric'], train_data['class'])

# Train SVM
print("Training SVM model...")
svm_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', LinearSVC(
        random_state=42,
        dual=False,  # Set explicitly to avoid future warning
        max_iter=10000,  # Increase max iterations to help convergence
        tol=1e-4  # Adjust tolerance to help convergence
    ))
])
svm_pipeline.fit(train_data['lyric'], train_data['class'])

# Get predictions 
print("Generating predictions...")
mnb_predictions = pd.DataFrame({
    'id': test_data['id'],
    'class': mnb_pipeline.predict(test_data['lyric'])
})

svm_predictions = pd.DataFrame({
    'id': test_data['id'],
    'class': svm_pipeline.predict(test_data['lyric'])
})

# Skip BERT for now as it takes longer to train
# We'll use placeholder predictions matching the ground truth size
print("Creating placeholder BERT predictions...")
bert_predictions = pd.DataFrame({
    'id': test_data['id'],
    'class': np.random.choice([0, 1], size=len(test_data), p=[0.57, 0.43])  # Approximate class distribution
})

# Save predictions
os.makedirs("src/results", exist_ok=True)
mnb_predictions.to_csv("src/results/mnb_predictions.csv", index=False)
svm_predictions.to_csv("src/results/svm_predictions.csv", index=False)
bert_predictions.to_csv("src/results/bert_predictions.csv", index=False)

print("Predictions saved to src/results/")

# Make sure the predictions and ground truth are aligned
print("Aligning predictions with ground truth...")
mnb_predictions = pd.merge(ground_truth[['id', 'class']], mnb_predictions, on='id', how='left', suffixes=('_true', ''))
svm_predictions = pd.merge(ground_truth[['id', 'class']], svm_predictions, on='id', how='left', suffixes=('_true', ''))
bert_predictions = pd.merge(ground_truth[['id', 'class']], bert_predictions, on='id', how='left', suffixes=('_true', ''))

# Print distribution after alignment
print(f"MNB predictions class distribution: {mnb_predictions['class'].value_counts().to_dict()}")
print(f"SVM predictions class distribution: {svm_predictions['class'].value_counts().to_dict()}")
print(f"BERT predictions class distribution: {bert_predictions['class'].value_counts().to_dict()}")

# Calculate performance metrics
print("Calculating performance metrics...")
mnb_accuracy = accuracy_score(ground_truth['class'], mnb_predictions['class'])
svm_accuracy = accuracy_score(ground_truth['class'], svm_predictions['class'])
bert_accuracy = accuracy_score(ground_truth['class'], bert_predictions['class'])

mnb_report = classification_report(ground_truth['class'], mnb_predictions['class'], output_dict=True, zero_division=0)
svm_report = classification_report(ground_truth['class'], svm_predictions['class'], output_dict=True, zero_division=0)
bert_report = classification_report(ground_truth['class'], bert_predictions['class'], output_dict=True, zero_division=0)

print(f"MNB Accuracy: {mnb_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"BERT Accuracy: {bert_accuracy:.4f}")

# Create comparative visualizations
print("Generating visualizations...")

# Class names for visualization
class_names = {0: 'Rap/Hip-Hop', 1: 'Pop'}

# 1. Accuracy comparison
plt.figure(figsize=(12, 6))
models = ['Multinomial Naive Bayes', 'Support Vector Machine', 'BERT (Placeholder)']
accuracies = [mnb_accuracy, svm_accuracy, bert_accuracy]
colors = ['#3498db', '#e74c3c', '#2ecc71']

plt.bar(models, accuracies, color=colors)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison on Test Data')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison_accuracy.png')
plt.close()

# 2. Confusion Matrix comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MNB confusion matrix
cm_mnb = confusion_matrix(ground_truth['class'], mnb_predictions['class'])
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[class_names[i] for i in sorted(class_names.keys())],
            yticklabels=[class_names[i] for i in sorted(class_names.keys())])
axes[0].set_title('MNB Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# SVM confusion matrix
cm_svm = confusion_matrix(ground_truth['class'], svm_predictions['class'])
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=[class_names[i] for i in sorted(class_names.keys())],
            yticklabels=[class_names[i] for i in sorted(class_names.keys())])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

# BERT confusion matrix
cm_bert = confusion_matrix(ground_truth['class'], bert_predictions['class'])
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[2],
            xticklabels=[class_names[i] for i in sorted(class_names.keys())],
            yticklabels=[class_names[i] for i in sorted(class_names.keys())])
axes[2].set_title('BERT Confusion Matrix (Placeholder)')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison_confusion.png')
plt.close()

# 3. Precision, Recall, F1 comparison
metrics = ['precision', 'recall', 'f1-score']
mnb_metrics = [mnb_report['weighted avg'][metric] for metric in metrics]
svm_metrics = [svm_report['weighted avg'][metric] for metric in metrics]
bert_metrics = [bert_report['weighted avg'][metric] for metric in metrics]

plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, mnb_metrics, width, label='MNB', color='#3498db')
plt.bar(x, svm_metrics, width, label='SVM', color='#e74c3c')
plt.bar(x + width, bert_metrics, width, label='BERT (Placeholder)', color='#2ecc71')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

# Add value labels
for i, v in enumerate(mnb_metrics):
    plt.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
for i, v in enumerate(svm_metrics):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
for i, v in enumerate(bert_metrics):
    plt.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison_metrics.png')
plt.close()

print(f"Comparison visualizations saved to {OUTPUT_DIR}")