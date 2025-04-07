import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import os
import pickle

MODEL_NAME = 'bert-base-uncased' 
MAX_LENGTH = 128 
BATCH_SIZE = 16 
EPOCHS = 3 
LEARNING_RATE = 5e-5 

TRAIN_DATA_PATH = 'src/data/source/train.csv'
TEST_DATA_PATH = 'src/data/target/test.csv'
RESULTS_DIR = 'src/results'
MODEL_SAVE_DIR = 'src/models/bert_fine_tuned' 
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, 'bert_predictions.csv')

def clean_text(text):
    """ Basic text cleaning. BERT tokenizers handle much of this, but consistency helps. """
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def load_and_preprocess_data(file_path, tokenizer, max_len, is_test=False):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if is_test:
        if 'lyric' not in df.columns:
            raise ValueError("Test CSV must contain 'lyric' column")
        df.dropna(subset=['lyric'], inplace=True)
        ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    else:
        if 'lyric' not in df.columns or 'class' not in df.columns:
            raise ValueError("Train CSV must contain 'lyric' and 'class' columns")
        df.dropna(subset=['lyric', 'class'], inplace=True)
        labels = df['class'].tolist()

    print("Cleaning text...")
    df['lyric_cleaned'] = df['lyric'].apply(clean_text)
    texts = df['lyric_cleaned'].tolist()

    print(f"Tokenizing text with '{MODEL_NAME}' tokenizer (max_len={max_len})...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='np') # Return numpy arrays directly

    if is_test:
        return dict(encodings), ids
    else:
        return dict(encodings), np.array(labels)


if __name__ == "__main__":
    print("--- TuneType: BERT Fine-Tuning ---")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_encodings_dict, all_labels = load_and_preprocess_data(
        TRAIN_DATA_PATH, tokenizer, MAX_LENGTH, is_test=False
    )

    num_samples = len(all_labels)
    if num_samples == 0:
        raise ValueError("No samples loaded from the training data.")
    
    for key, arr in all_encodings_dict.items():
        if arr.shape[0] != num_samples:
             raise ValueError(f"Inconsistent number of samples found in encoding '{key}'. Expected {num_samples}, got {arr.shape[0]}")
        print(f"Shape of '{key}': {arr.shape}")
    print(f"Shape of labels: {all_labels.shape}")

    print("Splitting data indices into training and validation sets (80/20)...")
    indices = np.arange(num_samples)

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )

    train_encodings_dict = {key: arr[train_indices] for key, arr in all_encodings_dict.items()}
    val_encodings_dict = {key: arr[val_indices] for key, arr in all_encodings_dict.items()}
    train_labels = all_labels[train_indices]
    val_labels = all_labels[val_indices]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings_dict, train_labels)).shuffle(len(train_labels)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_encodings_dict, val_labels)).batch(BATCH_SIZE)

    print(f"Training data size: {len(train_labels)}")
    print(f"Validation data size: {len(val_labels)}")

    print(f"Loading pre-trained model: {MODEL_NAME}")
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("Compiling model...")
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # 4. Fine-tune the Model
    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
    )
    print("Fine-tuning finished.")

    print("Evaluating final model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    print("Generating validation classification report...")
    val_predictions_output = model.predict(val_dataset)
    val_predictions_logits = val_predictions_output.logits
    val_predictions = np.argmax(val_predictions_logits, axis=1)
    print(classification_report(val_labels, val_predictions, target_names=['Rap/Hip-Hop (0)', 'Pop (1)']))

    print(f"Saving fine-tuned model and tokenizer to {MODEL_SAVE_DIR}...")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    if hasattr(history, 'history'):
        with open(os.path.join(MODEL_SAVE_DIR, 'bert.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        print("Training history saved.")
    else:
        print("Could not save training history.")
    print("Model and tokenizer saved.")


    print("Loading and preparing test data...")
    test_encodings_dict, test_ids = load_and_preprocess_data(
        TEST_DATA_PATH, tokenizer, MAX_LENGTH, is_test=True
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(test_encodings_dict).batch(BATCH_SIZE)

    print("Making predictions on test data...")
    test_predictions_output = model.predict(test_dataset)
    test_predictions_logits = test_predictions_output.logits
    test_predictions = np.argmax(test_predictions_logits, axis=1)

    print(f"Saving predictions to {PREDICTIONS_PATH}...")
    if len(test_ids) != len(test_predictions):
         print(f"Warning: Mismatch in length between test IDs ({len(test_ids)}) and predictions ({len(test_predictions)}). Cannot save CSV correctly.")
    else:
        results_df = pd.DataFrame({
            'id': test_ids,
            'class': test_predictions
        })
        results_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions saved to: {PREDICTIONS_PATH}")


    print("--- BERT Classification Complete ---")
    print(f"Fine-tuned model saved to: {MODEL_SAVE_DIR}")
