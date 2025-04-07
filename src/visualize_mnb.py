import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.feature_selection import mutual_info_classif
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
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 1))),  # Use only unigrams
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
plt.savefig('plots/mnb_class_distribution.png')
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
plt.savefig('plots/mnb_confusion_matrix.png')
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
plt.savefig('plots/mnb_roc_curve.png')
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
    plt.savefig(f'plots/mnb_top_features_{genre.lower().replace("/", "_")}.png')
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
plt.savefig('plots/mnb_precision_recall_f1.png')
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
plt.savefig('plots/mnb_accuracy.png')
plt.close()

# 7. Learning Curves
print("Generating learning curves...")
# Define the pipeline again to ensure clean training
pipeline_for_learning = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 1))),  # Use only unigrams
    ('classifier', MultinomialNB())
])

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    pipeline_for_learning, train_texts, train_labels, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curves (Multinomial Naive Bayes)', fontsize=16)
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)

# Add explanatory text
for i, (train_point, cv_point) in enumerate(zip(train_mean, test_mean)):
    if i == len(train_mean) - 1:  # Only annotate the last point
        plt.annotate(f'Training: {train_point:.3f}',
                     xy=(train_sizes[i], train_point),
                     xytext=(train_sizes[i]-500, train_point-0.03),
                     arrowprops=dict(arrowstyle='->'))
        plt.annotate(f'CV: {cv_point:.3f}',
                     xy=(train_sizes[i], cv_point),
                     xytext=(train_sizes[i]-500, cv_point+0.03),
                     arrowprops=dict(arrowstyle='->'))

# Add gap annotation
final_gap = train_mean[-1] - test_mean[-1]
plt.annotate(f'Gap: {final_gap:.3f}',
             xy=(train_sizes[-1], (train_mean[-1] + test_mean[-1])/2),
             xytext=(train_sizes[-1]*0.8, (train_mean[-1] + test_mean[-1])/2),
             arrowprops=dict(arrowstyle='<->'),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

plt.legend(loc="best", fontsize=12)
plt.tight_layout()
plt.savefig('plots/mnb_learning_curves.png', dpi=300)
plt.close()

# 8. Feature Importance Correlation
print("Generating feature importance correlation...")
# Get top features for each class
vectorizer = pipeline.named_steps['vectorizer']
classifier = pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

# Get the actual vocabulary from the trained vectorizer
original_vocabulary = vectorizer.vocabulary_

# Sort features by importance for each class
rap_log_probs = classifier.feature_log_prob_[0]
pop_log_probs = classifier.feature_log_prob_[1]
rap_indices = np.argsort(rap_log_probs)
pop_indices = np.argsort(pop_log_probs)

# Print top features for debugging
print("\nTop Rap features:")
for i in rap_indices[-20:]:
    print(f"{feature_names[i]}: {rap_log_probs[i]:.4f}")
    
print("\nTop Pop features:")
for i in pop_indices[-20:]:
    print(f"{feature_names[i]}: {pop_log_probs[i]:.4f}")

# Get top features from each genre
rap_features = []
pop_features = []

# Get top 8 features for each genre
for i in rap_indices[::-1]:
    feature = feature_names[i]
    if feature not in rap_features:
        rap_features.append(feature)
        if len(rap_features) >= 8:
            break

for i in pop_indices[::-1]:
    feature = feature_names[i]
    if feature not in pop_features:
        pop_features.append(feature)
        if len(pop_features) >= 8:
            break

# Print all selected features for debugging
print("\nSelected features for Rap:", rap_features)
print("Selected features for Pop:", pop_features)

# Combine features (handling duplicates)
all_features = rap_features + pop_features
selected_features = []
for feature in all_features:
    if feature not in selected_features:
        selected_features.append(feature)

print(f"Selected {len(selected_features)} unique top features for visualization")
print("Features for visualization:", selected_features)

# Validate that all features exist in the vocabulary
valid_features = []
for feature in selected_features:
    if feature in original_vocabulary:
        valid_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' not found in original vocabulary, skipping")

if len(valid_features) < 5:  # If we have too few valid features, use a manual fallback list
    print("Warning: Too few valid features, falling back to manually selected features")
    valid_features = ['love', 'baby', 'know', 'like', 'just', 'got', 'want', 'need', 'ain', 'yo']
    # Filter to make sure all these exist in vocabulary
    valid_features = [f for f in valid_features if f in original_vocabulary]

selected_features = valid_features
print(f"Final features for visualization: {len(selected_features)}")
print("Final feature list:", selected_features)

# Track which genre each feature belongs to
feature_genre = {}
for feature in selected_features:
    # Determine genre by checking their rank in each class
    rap_rank = -1 if feature not in [feature_names[i] for i in rap_indices[-50:]] else 1
    pop_rank = -1 if feature not in [feature_names[i] for i in pop_indices[-50:]] else 1
    
    # Assign genre based on where the feature ranks highly
    if rap_rank > 0 and pop_rank > 0:
        feature_genre[feature] = "Both"
    elif rap_rank > 0:
        feature_genre[feature] = "Rap"
    else:
        feature_genre[feature] = "Pop"

# Use the existing feature matrix from the pipeline's vectorizer
X_full = vectorizer.transform(train_texts)
feature_indices = [original_vocabulary[feature] for feature in selected_features]
X_selected = X_full[:, feature_indices]

# Convert to DataFrame for correlation analysis
df_features = pd.DataFrame(X_selected.toarray(), columns=selected_features)

# Calculate correlation matrix
correlation_matrix = df_features.corr()

# Check for NaN values or issues in the correlation matrix
if correlation_matrix.isnull().values.any():
    print("Warning: Correlation matrix contains NaN values. Replacing with zeros.")
    correlation_matrix = correlation_matrix.fillna(0)

# Create display names with genre labels
display_names = []
for name in selected_features:
    genre_label = feature_genre[name]
    display_name = f"{name} ({genre_label})"
    display_names.append(display_name)

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            annot=True,           # Show correlation values
            fmt=".2f",            # Format to 2 decimal places
            cmap='coolwarm',      # Red-blue color scheme
            vmin=-0.2, vmax=0.2,  # Limit color range to highlight smaller correlations
            square=True,          # Make cells square
            linewidths=0.5,       # Add cell borders
            xticklabels=display_names,
            yticklabels=display_names,
            annot_kws={"size": 9})  # Smaller font for annotations

# Print correlation matrix values for debugging
print("Correlation matrix shape:", correlation_matrix.shape)
print("Correlation matrix columns:", correlation_matrix.columns.tolist())

plt.title('Feature Correlation Matrix\nTop Features by Genre Importance', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add explanatory note about the correlation values
plt.figtext(0.5, 0.01, 
           "Correlation values range from -1 (never appear together) to +1 (always appear together)\n"
           "Values near zero indicate no relationship between word occurrences\n"
           "Features are labeled by genre importance: Rap, Pop, or Both",
           ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for annotation
plt.savefig('plots/mnb_feature_correlation.png', dpi=300)
plt.close()

# 9. Cross-Validation Performance
print("Generating cross-validation performance...")
# Perform k-fold cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, train_texts, train_labels, cv=k_fold, scoring='accuracy')

# Plot cross-validation performance
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue', width=0.6)
mean_acc = np.mean(cv_scores)
std_acc = np.std(cv_scores)
plt.axhline(y=mean_acc, color='red', linestyle='-', 
            label=f'Mean: {mean_acc:.4f} (±{std_acc:.4f})')

# Add min/max lines
plt.axhline(y=np.min(cv_scores), color='grey', linestyle='--', alpha=0.7,
            label=f'Min: {np.min(cv_scores):.4f}')
plt.axhline(y=np.max(cv_scores), color='grey', linestyle='--', alpha=0.7, 
            label=f'Max: {np.max(cv_scores):.4f}')

# Annotate the standard deviation range
plt.fill_between([0.5, len(cv_scores) + 0.5], mean_acc - std_acc, mean_acc + std_acc, 
                 color='red', alpha=0.1)

plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.title('5-Fold Cross-Validation Performance', fontsize=16)
plt.xticks(range(1, len(cv_scores) + 1))
plt.ylim(min(0.5, np.min(cv_scores) - 0.05), min(1.0, np.max(cv_scores) + 0.05))

# Add score labels above bars with color coding based on performance
for i, score in enumerate(cv_scores):
    color = 'green' if score >= mean_acc else 'red'
    plt.text(i + 1, score + 0.01, f'{score:.4f}', ha='center', fontsize=11, color=color)

# Add explanatory text
plt.figtext(0.5, 0.01, 
           "Consistent scores across folds indicate model stability\n"
           "Red shaded area shows ±1 standard deviation around the mean",
           ha="center", fontsize=12, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

plt.legend(loc="lower right", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig('plots/mnb_cross_validation.png', dpi=300)
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
results.to_csv('src/results/mnb_predictions.csv', index=False)
print("Predictions saved to src/results/mnb_predictions.csv")

print("Visualization complete! All plots saved to the 'plots' directory, including new advanced visualizations:")
print("- Learning curves showing model performance as training data increases")
print("- Feature correlation matrix showing relationships between top features")
print("- Cross-validation performance showing model stability across different data folds") 