# TuneType - Music Genre Classifier

A machine learning system for classifying music lyrics by genre (Rap/Hip-Hop vs Pop) using three approaches:

1. Multinomial Naive Bayes (MNB)
2. Support Vector Machine (SVM)
3. Bidirectional Encoder Representations from Transformers (Finetuned BERT)

## Setup

### Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- transformers (for BERT)
- tensorflow (for BERT)

### Environment Setup

1. **Create and activate virtual environment**:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Data Format

- **Training data**: CSV with 'lyric' and 'class' columns (0=Rap/Hip-Hop, 1=Pop)
- **Test data**: CSV with 'lyric' column and an 'id' column
- **Sample data**: Ground truth data for test set with 'id' and 'class' columns

## Running the Models

### Multinomial Naive Bayes

1. **Train and predict**:

   ```bash
   python src/mnb.py
   ```

This trains the model on training data and saves predictions to `src/results/mnb_predictions.csv`

2. **Generate visualizations**:
   ```bash
   python src/visualize_mnb.py
   ```

Creates visualizations in the `plots/` directory with the prefix 'mnb\_'.

### Support Vector Machine

1. **Train and predict**:

   ```bash
   python src/svm.py
   ```

This trains an SVM classifier, saves the model to `src/models/svm.pkl`, and predictions to `src/results/svm_predictions.csv`

2. **Generate visualizations**:
   ```bash
   python src/visualize_svm.py
   ```
   Creates visualizations in the `plots/` directory with the prefix 'svm\_'.

### Finetuned BERT

1. **Train and predict**:
   ```bash
   python src/bert.py
   ```

This fine-tunes a BERT model on the training data and saves predictions to `src/results/bert_predictions.csv`

2. **Generate visualizations**:
   ```bash
   python src/visualize_bert.py
   ```

Creates visualizations in the `plots/` directory with the prefix 'bert\_'.

## Model Evaluation

To compare the performance of all three models against the ground truth:

```bash
python src/eval.py
```

This will:

- Calculate accuracy, precision, recall, and F1-score for each model
- Generate comparative visualizations in the `plots/` directory
- Display accuracy results for all models

## Visualizations

### Model Visualizations

The visualization scripts generate:

- Class distribution
- Confusion matrix
- ROC curve
- Feature importance for each genre
- Precision, recall, and F1-score by class
- Overall accuracy

MNB additionally creates:

- Learning curves
- Feature correlation matrix
- Cross-validation performance

### Comparative Visualizations

The evaluation script generates:

- Comparative accuracy bar chart for all models
- Confusion matrices for all models
- Comparative precision, recall, and F1-score metrics

## Project Structure

```
├── plots/                       # Visualization outputs
├── src/
│   ├── data/
│   │   ├── source/
│   │   │   └── train.csv        # Training data (lyrics and labels)
│   │   └── target/
│   │       ├── sample.csv       # Ground truth for test data
│   │       └── test.csv         # Test data (lyrics only)
│   ├── models/
│   │   ├── bert_finetuned/      # BERT model files
│   │   └── svm.pkl              # Saved SVM model
│   ├── results/
│   │   ├── bert_predictions.csv # BERT predictions
│   │   ├── mnb_predictions.csv  # MNB predictions
│   │   └── svm_predictions.csv  # SVM predictions
│   ├── bert.py                  # BERT implementation
│   ├── eval.py                  # Model evaluation script
│   ├── mnb.py                   # MNB implementation
│   ├── svm.py                   # SVM implementation
│   ├── visualize_mnb.py         # Visualization for MNB results
│   └── visualize_svm.py         # Visualization for SVM results
│   └── visualize_bert.py         # Visualization for BERT results
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

## Notes

The BERT model is not in the `src/models/` folder due to file size constraints; it would have to be retrained.

The original dataset can be found [here](https://www.kaggle.com/datasets/sshikamaru/music-genre-classification/).
