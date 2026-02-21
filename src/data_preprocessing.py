"""
Data Preprocessing Module for Sentiment Analysis

This module handles all data preprocessing tasks including:
- Text cleaning
- Stopword removal (with negation preservation)
- Special character removal
- Stemming
- Tokenization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class DataPreprocessor:
    """Class for preprocessing text data for sentiment analysis"""
    
    def __init__(self):
        """Initialize the preprocessor with stopwords and stemmer"""
        # Initialize Porter Stemmer
        self.stemmer = PorterStemmer()
        
        # Define stopwords and customize to retain negation words
        default_stopwords = set(stopwords.words('english'))
        negation_words = {
            'not', 'no', "don't", "didn't", "won't", "couldn't", "wasn't", 
            'isn't', "aren't", "shouldn't", "can't", "haven't", "wouldn't", 
            "weren't", "mustn't", "mightn't", "shan't"
        }
        self.custom_stopwords = default_stopwords - negation_words
        
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords while preserving negation words
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        if isinstance(text, str):
            words = text.split()
            filtered_words = [word for word in words if word not in self.custom_stopwords]
            return ' '.join(filtered_words)
        return text
    
    def remove_special_characters(self, text):
        """
        Remove special characters except ! and ?
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with special characters removed
        """
        if isinstance(text, str):
            # Replace all special characters except ! and ? with a space
            return re.sub(r"[^a-zA-Z0-9!?]+", " ", text)
        return text
    
    def remove_extra_spaces(self, text):
        """
        Remove extra spaces from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized spacing
        """
        if isinstance(text, str):
            return ' '.join(text.split()).strip()
        return text
    
    def apply_stemming(self, text):
        """
        Apply Porter stemming to text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Stemmed text
        """
        if isinstance(text, str):
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        return text
    
    def apply_full_preprocessing(self, text):
        """
        Apply all preprocessing steps to text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Fully preprocessed text
        """
        text = self.preprocess_text(text)
        text = self.remove_stopwords(text)
        text = self.remove_special_characters(text)
        text = self.remove_extra_spaces(text)
        text = self.apply_stemming(text)
        return text


def load_and_preprocess_data(filepath, encoding='latin-1'):
    """
    Load dataset from CSV and apply preprocessing
    
    Args:
        filepath (str): Path to CSV file
        encoding (str): File encoding (default: 'latin-1')
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Load dataset
    data = pd.read_csv(filepath, encoding=encoding)
    
    print(f"Dataset loaded successfully: {data.shape[0]} rows")
    print(f"Columns: {data.columns.tolist()}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Apply preprocessing
    print("\nApplying preprocessing steps...")
    data['processed_text'] = data['Review'].apply(preprocessor.preprocess_text)
    data['processed_text'] = data['processed_text'].apply(preprocessor.remove_stopwords)
    data['processed_text'] = data['processed_text'].apply(preprocessor.remove_special_characters)
    data['processed_text'] = data['processed_text'].apply(preprocessor.remove_extra_spaces)
    data['processed_text'] = data['processed_text'].apply(preprocessor.apply_stemming)
    
    print("Preprocessing completed!")
    
    return data


def prepare_for_training(data, max_vocab_size=10000, max_seq_len=100, test_size=0.2, random_state=42):
    """
    Prepare data for model training
    
    Args:
        data (pd.DataFrame): Preprocessed dataset
        max_vocab_size (int): Maximum vocabulary size
        max_seq_len (int): Maximum sequence length
        test_size (float): Test set size ratio
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer, label_encoder)
    """
    # Encode labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['Polarity'])
    
    print(f"\nLabel encoding completed: {len(label_encoder.classes_)} classes")
    print(f"Classes: {label_encoder.classes_.tolist()}")
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['processed_text'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(data['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
    
    print(f"\nTokenization completed:")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Max sequence length: {max_seq_len}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, data['label'], test_size=test_size, random_state=random_state
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, tokenizer, label_encoder


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)
    
    # Load and preprocess data
    data = load_and_preprocess_data("updated_sentiment_analysis_dataset.csv")
    
    # Display sample
    print("\nSample processed data:")
    print(data[['Review', 'processed_text']].head())
    
    # Prepare for training
    X_train, X_test, y_train, y_test, tokenizer, label_encoder = prepare_for_training(data)