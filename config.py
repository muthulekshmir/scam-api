import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_default_secret_key'
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models/xgb_model.pkl'
    TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH') or 'models/distilbert_tokenizer'
    DEBUG = os.environ.get('DEBUG', 'False') == 'True'
    TESTING = os.environ.get('TESTING', 'False') == 'True'