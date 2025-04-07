"""
TuneType - SVM Classifier
Imesh Nimsitha
2025/04/06
"""

import os
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# FUNCTION: Clean lyrics text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', ' ', text.lower())  # remove special chars and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
        return text
    return ""

# FUNCTION: Load training data
def load_training_data(csv_path):
    df = pd.read_csv(csv_path)  # read CSV file
    
    # check required columns exist
    if 'lyric' not in df.columns or 'class' not in df.columns:
        raise ValueError("Training CSV file must contain 'lyric' and 'class' columns")

    df.dropna(subset=['lyric', 'class'], inplace=True)  # drop rows with missing values
    
    df['lyric'] = df['lyric'].apply(clean_text)  # clean the text
    
    texts = df['lyric'].tolist()  # extract texts and labels
    labels = df['class'].tolist()
    return texts, labels

# FUNCTION: Load test data
def load_test_data(csv_path):
    df = pd.read_csv(csv_path)  # read CSV file
    
    # check lyric column exists
    if 'lyric' not in df.columns:
        raise ValueError("Test CSV file must contain a 'lyric' column")

    df['lyric'] = df['lyric'].apply(clean_text)  # clean the text
    
    # get IDs (or create indices if no ID column)
    ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    texts = df['lyric'].tolist()
        
    return ids, texts

# FUNCTION: Train SVM model
def train_model(csv_path, test_size=0.2, random_state=42, model_path=None):
    # load training dataset
    texts, labels = load_training_data(csv_path)
    
    # split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    
    # vectorize with TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.95, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # initialize and train SVM
    svm = LinearSVC(C=5.0, class_weight='balanced')
    svm.fit(X_train_tfidf, y_train)
    
    # validate model
    y_pred = svm.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    print("=== TuneType - SVM Classifier Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # retrain on full dataset
    print("Retraining on the entire dataset...")
    X_all_tfidf = vectorizer.fit_transform(texts)
    svm.fit(X_all_tfidf, labels)
    
    # save model if path provided
    if model_path:
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        with open(model_path, 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'model': svm}, f)
        print(f"Model saved to {model_path}")
    
    return vectorizer, svm

# FUNCTION: Test model on new data
def test_model(csv_path, model_path, output_dir='src/results'):
    # load saved model and vectorizer
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    vectorizer = saved['vectorizer']
    svm = saved['model']
    
    # load and process test data
    ids, texts = load_test_data(csv_path)
    X_tfidf = vectorizer.transform(texts)
    
    # generate predictions
    y_pred = svm.predict(X_tfidf)
    
    # create results dataframe
    results_df = pd.DataFrame({
        'id': ids,
        'class': y_pred
    })
    
    # save predictions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'svm_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return results_df

# MAIN
def main():
    train_csv_path = "src/data/source/train.csv"
    test_csv_path = "src/data/target/test.csv"
    model_path = "src/models/svm.pkl"
    output_dir = "src/results"
    test_size = 0.2
    random_state = 42
    
    try:
        # train the model
        print("Training SVM model...")
        train_model(train_csv_path, test_size, random_state, model_path)
        
        # test the model
        print("Testing model on new data...")
        test_model(test_csv_path, model_path, output_dir)
        print("Done! Model saved to", model_path)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()