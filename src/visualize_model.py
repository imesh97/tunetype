import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import re
import os

# Create directory for visualizations
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_dataset(csv_path, is_test=False):
    """
    Loads the dataset from a CSV file.

    Expects a CSV file with at least two columns:
      - 'lyric': The text data (lyrics)
      - 'class': The corresponding class labels (not required for test data)

    Parameters:
        csv_path (str): Path to the CSV file.
        is_test (bool): Whether this is a test dataset without class labels.

    Returns:
        texts (list): List of lyric strings.
        labels (list): List of corresponding labels (or None for test data).
    """
    df = pd.read_csv(csv_path)
    
    # For test data, we only need the lyric column
    if is_test:
        if 'lyric' not in df.columns:
            raise ValueError("CSV file must contain a 'lyric' column")
        df.dropna(subset=['lyric'], inplace=True)
        texts = df['lyric'].tolist()
        return texts, None
    
    # For training data, we need both lyric and class columns
    if 'lyric' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV file must contain 'lyric' and 'class' columns")

    # Optionally, drop rows with missing values
    df.dropna(subset=['lyric', 'class'], inplace=True)
    
    texts = df['lyric'].tolist()
    labels = df['class'].tolist()
    return texts, labels

# Function to clean lyrics text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Load and preprocess data
print("Loading data...")
train_texts, train_labels = load_dataset('src/data/source/train.csv')

# Clean the texts
train_texts = [clean_text(text) for text in train_texts]

# Class names
class_names = {0: 'Rap/Hip-Hop', 1: 'Pop'}

# Create a DataFrame for easier plotting
train_data = pd.DataFrame({
    'lyric': train_texts,
    'class': train_labels,
    'genre': [class_names[label] for label in train_labels]
})

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, 
    test_size=0.2, random_state=42, stratify=train_labels
)

# Create and train model
print("Training model...")
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Get predictions
y_pred = pipeline.predict(X_val)
y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

# 1. Plot class distribution
print("Generating class distribution chart...")
plt.figure(figsize=(10, 6))
# Count instances of each class
class_count_dict = {class_names[0]: train_labels.count(0), class_names[1]: train_labels.count(1)}
class_names_list = list(class_count_dict.keys())
class_counts_list = list(class_count_dict.values())

plt.bar(class_names_list, class_counts_list, color=['#3498db', '#e74c3c'])
plt.title('Distribution of Classes in Training Data', fontsize=16)
plt.xlabel('Genre')
plt.ylabel('Count')
for i, count in enumerate(class_counts_list):
    plt.text(i, count + 200, f'{count}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('plots/class_distribution.png')
plt.close()

# 2. Plot confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[class_names[i] for i in range(2)],
            yticklabels=[class_names[i] for i in range(2)])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()

# 3. Plot ROC curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plots/roc_curve.png')
plt.close()

# 4. Plot feature importance
print("Generating feature importance plot...")
vectorizer = pipeline.named_steps['vectorizer']
classifier = pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

# Get top features for both classes
for i, genre in class_names.items():
    # Sort features by their importance for this class
    log_probs = classifier.feature_log_prob_[i]
    top_indices = np.argsort(log_probs)[-15:]
    top_features = [feature_names[j] for j in top_indices]
    top_probs = [log_probs[j] for j in top_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_probs, align='center', color='teal')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Log Probability')
    plt.title(f'Top 15 Features for {genre}')
    plt.tight_layout()
    plt.savefig(f'plots/top_features_{genre.lower().replace("/", "_")}.png')
    plt.close()

# 5. Plot precision-recall by class
print("Generating precision-recall plot...")
# Fixed the classification report to use numeric labels
report = classification_report(y_val, y_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
class_labels = ['0', '1']  # Use string representations of class numbers as keys
values = [[report[label][metric] for metric in metrics] for label in class_labels]

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, values[0], width, label=class_names[0], color='royalblue')
plt.bar(x + width/2, values[1], width, label=class_names[1], color='lightcoral')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score by Class')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/precision_recall_f1.png')
plt.close()

# 6. Accuracy Bar Chart
print("Generating accuracy chart...")
accuracy = accuracy_score(y_val, y_pred)
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='green', width=0.5)
plt.title('Model Accuracy', fontsize=16)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('plots/accuracy.png')
plt.close()

# Also predict on test data
print("Generating predictions for test data...")
test_texts, _ = load_dataset('src/data/target/test.csv', is_test=True)
test_texts = [clean_text(text) for text in test_texts]
test_predictions = pipeline.predict(test_texts)

# Create DataFrame for predictions
test_df = pd.read_csv('src/data/target/test.csv')
results = pd.DataFrame({
    'id': test_df['id'],
    'class': test_predictions
})
results.to_csv('src/data/target/MNB_predictions.csv', index=False)
print("Predictions saved to src/data/target/MNB_predictions.csv")

print("Visualization complete! All plots saved to the 'plots' directory.") 