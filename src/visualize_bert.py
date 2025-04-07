import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import re
import os
import pickle
import sys

MODEL_SAVE_DIR = 'src/models/bert_fine_tuned/' 
TRAIN_DATA_PATH = 'src/data/source/train.csv'
PLOTS_DIR = 'plots'
MAX_LENGTH = 128 
BATCH_SIZE = 16 

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def load_and_tokenize_for_validation(csv_path, tokenizer, max_len):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if 'lyric' not in df.columns or 'class' not in df.columns:
        raise ValueError("Train CSV must contain 'lyric' and 'class' columns")
    df.dropna(subset=['lyric', 'class'], inplace=True)

    print("Cleaning text...")
    df['lyric_cleaned'] = df['lyric'].apply(clean_text)
    texts = df['lyric_cleaned'].tolist()
    labels = df['class'].tolist()

    print("Splitting data into training and validation sets (80/20) for evaluation consistency...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Tokenizing validation text with tokenizer (max_len={max_len})...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len, return_tensors='tf')

    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(BATCH_SIZE)

    return val_dataset, val_labels, train_labels # Return train_labels for class distribution plot

if __name__ == "__main__":
    print("--- TuneType: BERT Model Visualizer ---")

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    
    if not os.path.exists(MODEL_SAVE_DIR):
        print("Can't find dir bro")
        sys.exit(1)

    if not os.path.exists(MODEL_SAVE_DIR) or \
       not os.path.exists(os.path.join(MODEL_SAVE_DIR, 'tf_model.h5')) or \
       not os.path.exists(os.path.join(MODEL_SAVE_DIR, 'tokenizer_config.json')):
        print(f"Error: Fine-tuned BERT model not found in {MODEL_SAVE_DIR}")
        print("Please run the bert.py script first to train and save the model.")
        sys.exit(1)

    print(f"Loading fine-tuned model and tokenizer from {MODEL_SAVE_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    try:
        val_dataset, y_val, train_labels_full = load_and_tokenize_for_validation(
            TRAIN_DATA_PATH, tokenizer, MAX_LENGTH
        )
        print(f"Validation set prepared with {len(y_val)} samples.")
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        sys.exit(1)

    print("Generating predictions for the validation set...")
    try:
        predictions_output = model.predict(val_dataset)
        logits = predictions_output.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        y_pred = np.argmax(probabilities, axis=1)
        y_pred_proba = probabilities[:, 1]
        print("Predictions generated.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    class_names = {0: 'Rap/Hip-Hop', 1: 'Pop'} 

    print("Generating class distribution chart...")
    plt.figure(figsize=(10, 6))
    class_count_dict = {class_names[0]: train_labels_full.count(0), class_names[1]: train_labels_full.count(1)}
    class_names_list = list(class_count_dict.keys())
    class_counts_list = list(class_count_dict.values())
    plt.bar(class_names_list, class_counts_list, color=['#3498db', '#e74c3c'])
    plt.title('Distribution of Classes in Training Data', fontsize=16)
    plt.xlabel('Genre')
    plt.ylabel('Count')
    for i, count in enumerate(class_counts_list):
        plt.text(i, count + 200, f'{count}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bert_class_distribution.png'))
    plt.close()
    print("Class distribution chart saved.")

    # VISUALIZATION 2: Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in range(2)],
                yticklabels=[class_names[i] for i in range(2)])
    plt.title('BERT Confusion Matrix (Validation Set)', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bert_confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved.")

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
    plt.title('BERT Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bert_roc_curve.png'))
    plt.close()
    print("ROC curve saved.")

    print("Generating precision-recall plot...")
    report = classification_report(y_val, y_pred, output_dict=True, target_names=class_names.values())
    metrics = ['precision', 'recall', 'f1-score']
    report_class_keys = [key for key in report.keys() if key in class_names.values()]
    values = [[report[label][metric] for metric in metrics] for label in report_class_keys]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    rap_index = report_class_keys.index(class_names[0])
    pop_index = report_class_keys.index(class_names[1])
    plt.bar(x - width/2, values[rap_index], width, label=class_names[0], color='royalblue')
    plt.bar(x + width/2, values[pop_index], width, label=class_names[1], color='lightcoral')
    plt.ylabel('Score')
    plt.title('BERT Precision, Recall, and F1-Score by Class (Validation Set)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bert_precision_recall_f1.png'))
    plt.close()
    print("Precision-recall plot saved.")

    print("Generating accuracy chart...")
    accuracy = accuracy_score(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy], color='green', width=0.5)
    plt.title('BERT Model Accuracy (Validation Set)', fontsize=16)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bert_accuracy.png'))
    plt.close()
    print("Accuracy chart saved.")

    history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.pkl')
    if os.path.exists(history_path):
        print("Generating training history plots...")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        epochs_range = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        acc_key = 'accuracy' if 'accuracy' in history else 'sparse_categorical_accuracy'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_sparse_categorical_accuracy'

        if acc_key in history:
             plt.plot(epochs_range, history[acc_key], label='Training Accuracy')
        if val_acc_key in history:
             plt.plot(epochs_range, history[val_acc_key], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'bert_training_history.png'))
        plt.close()
        print("Training history plots saved.")
    else:
        print("Training history file not found, skipping history plots.")


    print("\n--- BERT Visualization Complete ---")
    print(f"All plots saved to the '{PLOTS_DIR}' directory with 'bert_' prefix.")
    print("\nValidation Performance Summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report (Validation Set):")
    print(classification_report(y_val, y_pred, target_names=[class_names[0], class_names[1]]))