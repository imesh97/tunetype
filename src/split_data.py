"""
TuneType - Data Splitter
Creates a proper 80/20 train/test split from the training data
for accurate model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Paths
TRAIN_DATA_PATH = "src/data/source/train.csv"
OUTPUT_DIR = "src/data/split"
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_split.csv")
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_split.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load training data
print("Loading training data...")
train_data = pd.read_csv(TRAIN_DATA_PATH)

# Check class distribution in original data
class_counts = train_data['class'].value_counts()
print(f"Original class distribution:\n{class_counts}")
print(f"Total samples: {len(train_data)}")

# Create train/test split with stratification to maintain class balance
print("Creating 80/20 train/test split with stratification...")
train_df, test_df = train_test_split(
    train_data, 
    test_size=0.2,
    random_state=42,
    stratify=train_data['class']  # Ensures class balance is maintained
)

# Verify the split
print(f"Training set size: {len(train_df)} samples")
print(f"Test set size: {len(test_df)} samples")
print(f"Training class distribution:\n{train_df['class'].value_counts()}")
print(f"Test class distribution:\n{test_df['class'].value_counts()}")

# Save the split datasets
print(f"Saving training split to {TRAIN_OUTPUT_PATH}")
train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)

print(f"Saving test split to {TEST_OUTPUT_PATH}")
test_df.to_csv(TEST_OUTPUT_PATH, index=False)

print("Data split complete!") 