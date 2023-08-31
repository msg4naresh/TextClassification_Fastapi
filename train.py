import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from text_preprocessor import TextPreprocessor


import pickle
import logging

from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(str(parent_dir))

# Load Data
try:
    true_df = pd.read_csv(str(current_dir)+"/data/True.csv")
    fake_df = pd.read_csv(str(current_dir)+"/data/Fake.csv")
    true_df['label'] = 'True'
    fake_df['label'] = 'Fake'
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    logger.info("Data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading data: {e}")

combined_df['content'] = combined_df['title'] + ' ' + combined_df['text']

# Preprocessing and Feature Extraction
X = combined_df['content']
y = combined_df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('text_preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Model Training
try:
    pipeline.fit(X_train, y_train)
    logger.info("Model trained successfully.")
except Exception as e:
    logger.error(f"Error during model training: {e}")

# Save Model
try:
    with open(str(current_dir) + "/model.pkl", 'wb') as f: 
        pickle.dump(pipeline, f)
    logger.info("Model saved successfully.")
except Exception as e:
    logger.error(f"Error saving model: {e}")
