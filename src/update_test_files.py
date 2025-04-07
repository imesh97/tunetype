"""
TuneType - Test Files Updater
Updates sample.csv and test.csv with the contents of test_split.csv
to ensure proper model evaluation
"""

import pandas as pd
import os
import shutil

# Define paths
TEST_SPLIT_PATH = "src/data/target/test_split.csv"
SAMPLE_PATH = "src/data/target/sample.csv"
TEST_PATH = "src/data/target/test.csv"

# Create backups
print("Creating backups of original files...")
shutil.copy2(SAMPLE_PATH, f"{SAMPLE_PATH}.bak")
shutil.copy2(TEST_PATH, f"{TEST_PATH}.bak")
print("Backups created.")

# Load the test split data
print(f"Loading test split data from {TEST_SPLIT_PATH}...")
test_split_data = pd.read_csv(TEST_SPLIT_PATH)

# Add id column if not present
if 'id' not in test_split_data.columns:
    print("Adding id column to test data...")
    test_split_data['id'] = range(len(test_split_data))

# Prepare data for sample.csv (id and class only)
print("Preparing data for sample.csv (id and class only)...")
sample_data = test_split_data.copy()
if 'class' in sample_data.columns:
    sample_data = sample_data[['id', 'class']]
else:
    print("Warning: 'class' column not found in test split data!")

# Prepare data for test.csv (id and lyric only)
print("Preparing data for test.csv (id and lyric only)...")
test_data = test_split_data.copy()
if 'lyric' in test_data.columns:
    test_data = test_data[['id', 'lyric']]
else:
    print("Warning: 'lyric' column not found in test split data!")

# Save the prepared data to sample.csv and test.csv
print(f"Saving updated data to {SAMPLE_PATH}...")
sample_data.to_csv(SAMPLE_PATH, index=False)

print(f"Saving updated data to {TEST_PATH}...")
test_data.to_csv(TEST_PATH, index=False)

# Verify the updates
sample_file_size = os.path.getsize(SAMPLE_PATH)
test_file_size = os.path.getsize(TEST_PATH)

print(f"Update complete!")
print(f"New sample.csv size: {sample_file_size} bytes")
print(f"New test.csv size: {test_file_size} bytes")

# Verify contents of both files
sample_df = pd.read_csv(SAMPLE_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Sample.csv columns: {', '.join(sample_df.columns)}")
print(f"Test.csv columns: {', '.join(test_df.columns)}")

# Print class distribution in sample.csv
if 'class' in sample_df.columns:
    class_counts = sample_df['class'].value_counts()
    print(f"Class distribution in sample.csv:")
    print(class_counts) 