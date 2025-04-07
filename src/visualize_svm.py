"""
TuneType - SVM Model Visualizer
Imesh Nimsitha
2025/04/06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import re
import os
import pickle
import sys

# create visualization directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# FUNCTION: Load dataset from CSV
def load_dataset(csv_path, is_test=False):
    df = pd.read_csv(csv_path)  # read CSV file
    
    if is_test:  # handling test data
        if 'lyric' not in df.columns:
            raise ValueError("CSV file must contain a 'lyric' column")
        df.dropna(subset=['lyric'], inplace=True)
        texts = df['lyric'].tolist()
        return texts, None
    
    # handling training data
    if 'lyric' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV file must contain 'lyric' and 'class' columns")

    df.dropna(subset=['lyric', 'class'], inplace=True)  # drop rows with missing values
    
    texts = df['lyric'].tolist()  # extract texts and labels
    labels = df['class'].tolist()
    return texts, labels

# FUNCTION: Clean lyrics text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', ' ', text.lower())  # remove special chars and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
        return text
    return ""

# check if SVM model exists
model_path = 'src/models/svm.pkl'
if not os.path.exists(model_path):
    print("Error: SVM model not found at src/models/svm.pkl")
    print("Please run the svm.py script first.")
    sys.exit(1)

# load pre-trained model
print("Loading pre-trained SVM model...")
with open(model_path, 'rb') as f:
    saved = pickle.load(f)
    vectorizer = saved['vectorizer']  # extract vectorizer and model
    model = saved['model']

# load and preprocess training data
print("Loading and preprocessing data...")
train_texts, train_labels = load_dataset('src/data/source/train.csv')
train_texts = [clean_text(text) for text in train_texts]  # clean texts

# define class names for visualization
class_names = {0: 'Rap/Hip-Hop', 1: 'Pop'}

# create dataframe for easier plotting
train_data = pd.DataFrame({
    'lyric': train_texts,
    'class': train_labels,
    'genre': [class_names[label] for label in train_labels]
})

# split data for validation (using same random_state as training)
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, 
    test_size=0.2, random_state=42, stratify=train_labels
)

# generate predictions for validation
print("Generating predictions for validation set...")
X_val_tfidf = vectorizer.transform(X_val)  # vectorize validation texts
y_pred = model.predict(X_val_tfidf)  # predict classes

# prepare data for ROC curve
y_decision = model.decision_function(X_val_tfidf)  # get decision scores
y_decision_scaled = (y_decision - np.min(y_decision)) / (np.max(y_decision) - np.min(y_decision))  # scale to [0,1]

# VISUALIZATION 1: Class distribution
print("Generating class distribution chart...")
plt.figure(figsize=(10, 6))
class_count_dict = {class_names[0]: train_labels.count(0), class_names[1]: train_labels.count(1)}  # count classes
class_names_list = list(class_count_dict.keys())
class_counts_list = list(class_count_dict.values())

plt.bar(class_names_list, class_counts_list, color=['#3498db', '#e74c3c'])  # create bar chart
plt.title('Distribution of Classes in Training Data', fontsize=16)
plt.xlabel('Genre')
plt.ylabel('Count')
for i, count in enumerate(class_counts_list):  # add count labels
    plt.text(i, count + 200, f'{count}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('plots/svm_class_distribution.png')
plt.close()

# VISUALIZATION 2: Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_val, y_pred)  # compute confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # create heatmap
            xticklabels=[class_names[i] for i in range(2)],
            yticklabels=[class_names[i] for i in range(2)])
plt.title('SVM Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('plots/svm_confusion_matrix.png')
plt.close()

# VISUALIZATION 3: ROC curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_val, y_decision_scaled)  # compute ROC curve
roc_auc = auc(fpr, tpr)  # compute AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('SVM Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plots/svm_roc_curve.png')
plt.close()

# VISUALIZATION 4: Feature importance
print("Generating feature importance plot...")
feature_names = vectorizer.get_feature_names_out()  # get feature names
coefficients = model.coef_[0]  # get SVM coefficients

# Plot top features for Rap (most negative coefficients)
rap_indices = np.argsort(coefficients)[:15]  # find top 15 negative coefficients
rap_features = [feature_names[i] for i in rap_indices]
rap_coeffs = [coefficients[i] for i in rap_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(rap_features)), rap_coeffs, align='center', color='royalblue')  # horizontal bar chart
plt.yticks(range(len(rap_features)), rap_features)
plt.xlabel('Coefficient Value')
plt.title(f'Top 15 Features for {class_names[0]} (SVM)')
plt.tight_layout()
plt.savefig(f'plots/svm_top_features_{class_names[0].lower().replace("/", "_")}.png')
plt.close()

# Plot top features for Pop (most positive coefficients)
pop_indices = np.argsort(coefficients)[-15:][::-1]  # find top 15 positive coefficients
pop_features = [feature_names[i] for i in pop_indices]
pop_coeffs = [coefficients[i] for i in pop_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(pop_features)), pop_coeffs, align='center', color='lightcoral')  # horizontal bar chart
plt.yticks(range(len(pop_features)), pop_features)
plt.xlabel('Coefficient Value')
plt.title(f'Top 15 Features for {class_names[1]} (SVM)')
plt.tight_layout()
plt.savefig(f'plots/svm_top_features_{class_names[1].lower().replace("/", "_")}.png')
plt.close()

# VISUALIZATION 5: Precision-recall by class
print("Generating precision-recall plot...")
report = classification_report(y_val, y_pred, output_dict=True)  # get classification metrics
metrics = ['precision', 'recall', 'f1-score']
class_labels = ['0', '1']  # use string representations of class numbers
values = [[report[label][metric] for metric in metrics] for label in class_labels]

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, values[0], width, label=class_names[0], color='royalblue')  # group bar chart
plt.bar(x + width/2, values[1], width, label=class_names[1], color='lightcoral')
plt.ylabel('Score')
plt.title('SVM Precision, Recall, and F1-Score by Class')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/svm_precision_recall_f1.png')
plt.close()

# VISUALIZATION 6: Accuracy Bar Chart
print("Generating accuracy chart...")
accuracy = accuracy_score(y_val, y_pred)  # compute accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='green', width=0.5)  # create bar chart
plt.title('SVM Model Accuracy', fontsize=16)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center', fontsize=12)  # add accuracy value
plt.tight_layout()
plt.savefig('plots/svm_accuracy.png')
plt.close()

# check if predictions file exists
predictions_path = 'src/results/svm_predictions.csv'
if not os.path.exists(predictions_path):
    print("Warning: Predictions file not found at src/results/svm_predictions.csv")
    print("The classifier may not have been run properly.")

print("Visualization complete! All plots saved to the 'plots' directory with 'svm_' prefix.")