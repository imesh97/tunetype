"""
TuneType - Model Evaluation
Compares model predictions against ground truth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths
ground_truth_path = "src/data/target/sample.csv"
mnb_predictions_path = "src/results/mnb_predictions.csv"
svm_predictions_path = "src/results/svm_predictions.csv"
bert_predictions_path = "src/results/bert_predictions.csv"  # Added BERT predictions path
output_dir = "plots"

# Load data
ground_truth = pd.read_csv(ground_truth_path)
mnb_predictions = pd.read_csv(mnb_predictions_path)
svm_predictions = pd.read_csv(svm_predictions_path)
bert_predictions = pd.read_csv(bert_predictions_path)  # Load BERT predictions

# Ensure data is properly aligned (match by id)
ground_truth = ground_truth.sort_values('id')
mnb_predictions = mnb_predictions.sort_values('id')
svm_predictions = svm_predictions.sort_values('id')
bert_predictions = bert_predictions.sort_values('id')  # Sort BERT predictions

# Verify that the ids match
if not (ground_truth['id'].equals(mnb_predictions['id']) and 
        ground_truth['id'].equals(svm_predictions['id']) and
        ground_truth['id'].equals(bert_predictions['id'])):
    print("Warning: IDs in prediction files don't match ground truth exactly.")
    # Align datasets by ID
    mnb_predictions = pd.merge(ground_truth[['id']], mnb_predictions, on='id', how='left')
    svm_predictions = pd.merge(ground_truth[['id']], svm_predictions, on='id', how='left')
    bert_predictions = pd.merge(ground_truth[['id']], bert_predictions, on='id', how='left')

# Class names for visualization
class_names = {0: 'Rap/Hip-Hop', 1: 'Pop'}

# Calculate performance metrics
mnb_accuracy = accuracy_score(ground_truth['class'], mnb_predictions['class'])
svm_accuracy = accuracy_score(ground_truth['class'], svm_predictions['class'])
bert_accuracy = accuracy_score(ground_truth['class'], bert_predictions['class'])  # Calculate BERT accuracy

mnb_report = classification_report(ground_truth['class'], mnb_predictions['class'], output_dict=True, zero_division=0)
svm_report = classification_report(ground_truth['class'], svm_predictions['class'], output_dict=True, zero_division=0)
bert_report = classification_report(ground_truth['class'], bert_predictions['class'], output_dict=True, zero_division=0)  # BERT report

print(f"MNB Accuracy: {mnb_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"BERT Accuracy: {bert_accuracy:.4f}")  # Print BERT accuracy

# Create comparative visualizations

# 1. Accuracy comparison
plt.figure(figsize=(12, 6))
models = ['Multinomial Naive Bayes', 'Support Vector Machine', 'BERT']  # Add BERT to models
accuracies = [mnb_accuracy, svm_accuracy, bert_accuracy]  # Add BERT accuracy
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Add color for BERT

plt.bar(models, accuracies, color=colors)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison on Test Data')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/model_comparison_accuracy.png')
plt.close()

# 2. Confusion Matrix comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Expand to 3 columns for BERT

# MNB confusion matrix
cm_mnb = confusion_matrix(ground_truth['class'], mnb_predictions['class'])
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[class_names[i] for i in range(2)],
            yticklabels=[class_names[i] for i in range(2)])
axes[0].set_title('MNB Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# SVM confusion matrix
cm_svm = confusion_matrix(ground_truth['class'], svm_predictions['class'])
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=[class_names[i] for i in range(2)],
            yticklabels=[class_names[i] for i in range(2)])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

# BERT confusion matrix
cm_bert = confusion_matrix(ground_truth['class'], bert_predictions['class'])
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[2],  # Add BERT confusion matrix
            xticklabels=[class_names[i] for i in range(2)],
            yticklabels=[class_names[i] for i in range(2)])
axes[2].set_title('BERT Confusion Matrix')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

plt.tight_layout()
plt.savefig(f'{output_dir}/model_comparison_confusion.png')
plt.close()

# 3. Precision, Recall, F1 comparison
metrics = ['precision', 'recall', 'f1-score']
mnb_metrics = [mnb_report['weighted avg'][metric] for metric in metrics]
svm_metrics = [svm_report['weighted avg'][metric] for metric in metrics]
bert_metrics = [bert_report['weighted avg'][metric] for metric in metrics]  # Add BERT metrics

plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.25  # Narrower bars to fit three models

plt.bar(x - width, mnb_metrics, width, label='MNB', color='#3498db')
plt.bar(x, svm_metrics, width, label='SVM', color='#e74c3c')
plt.bar(x + width, bert_metrics, width, label='BERT', color='#2ecc71')  # Add BERT bars
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
plt.savefig(f'{output_dir}/model_comparison_metrics.png')
plt.close()

print(f"Comparison visualizations saved to {output_dir}")