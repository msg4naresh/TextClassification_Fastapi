import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK's stop words and WordNetLemmatizer
stop_words_nltk = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom list of stop words
stop_words_custom = set(["werrt"])

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = stop_words_nltk.union(stop_words_custom)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X_transformed = [self.preprocess_text(text) for text in X]
            return X_transformed
        except Exception as e:
            logger.error(f"Error occurred during text transformation: {e}")
            return []

    def preprocess_text(self, text):
        """Execute text preprocessing steps."""
        try:
            text = self.to_lower_case(text)
            text = self.remove_punctuation(text)
            text = self.remove_numbers(text)
            text = self.lemmatize_words(text)
            text = self.remove_stop_words(text)
            return text
        except Exception as e:
            logger.error(f"Error occurred during preprocessing individual text: {e}")
            return ""

    def to_lower_case(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def lemmatize_words(self, text):
        word_tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(w) for w in word_tokens])

    def remove_stop_words(self, text):
        word_tokens = word_tokenize(text)
        return ' '.join([w for w in word_tokens if w not in self.stop_words])

# Test the TextPreprocessor class
if __name__ == "__main__":
    sample_text = "This is an example sentence werrt to test the TextPreprocessor class."
    tp = TextPreprocessor()
    processed_text = tp.transform([sample_text])
    print(f"Original text: {sample_text}")
    print(f"Processed text: {processed_text[0]}")
