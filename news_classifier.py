import pickle
import logging
from text_preprocessor import TextPreprocessor
from pathlib import Path
# Get the directory of the current file
current_dir = Path(__file__).parent


logging.basicConfig(level=logging.INFO)
logger =logging.getLogger(__name__)


class NewsClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsClassifier,cls).__new__(cls)
            #Load model when the application starts
            try:
                cls._instance.model=pickle.load(open("model.pkl", 'rb'))
                logger.info("Model Loaded Successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
            cls._instance.text_preprocessor = TextPreprocessor()
        return cls._instance
    

    
    def predict(self,var):
        try:
            self.validate_input(var)
            var = self._instance.text_preprocessor.transform([var])[0]
            return self.model.predict([var])[0]
        except Exception as e:
            logger.error(f"An error occured  during  prediction: {e}")
            return "Error During Prediction"
        
    def validate_input(self,var):
        if not var:
            raise ValueError("Input is empty")
        if len(var)>5000:
            raise ValueError("Input exceeds maximum length")
        if not isinstance(var,str):
            raise ValueError("Input should be a string")
        
# create a singleton Instance
news_classifier = NewsClassifier()

def predict_news(var):
    return news_classifier.predict(var)




