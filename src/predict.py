"""
Prediction Script for Sentiment Analysis

This script loads a trained model and makes predictions on new text
"""

import os
import sys
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor


class SentimentPredictor:
    """Class for predicting sentiment from text"""
    
    def __init__(self, model_path, tokenizer_path, label_encoder_path, max_seq_len=100):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to saved model
            tokenizer_path (str): Path to saved tokenizer
            label_encoder_path (str): Path to saved label encoder
            max_seq_len (int): Maximum sequence length
        """
        # Load model
        self.model = load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from: {tokenizer_path}")
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from: {label_encoder_path}")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        self.max_seq_len = max_seq_len
        
        # Get sentiment labels
        self.sentiment_labels = self.label_encoder.classes_.tolist()
        print(f"Classes: {self.sentiment_labels}")
    
    def preprocess_text(self, text):
        """
        Preprocess text for prediction
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        return self.preprocessor.apply_full_preprocessing(text)
    
    def predict(self, text, return_proba=False):
        """
        Predict sentiment for text
        
        Args:
            text (str): Input text
            return_proba (bool): Whether to return probabilities
        
        Returns:
            str or tuple: Predicted sentiment (and probabilities if return_proba=True)
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_seq_len, padding='post')
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction)
        predicted_sentiment = self.sentiment_labels[predicted_class_idx]
        
        if return_proba:
            # Return sentiment with probabilities
            probabilities = {
                label: float(prob) 
                for label, prob in zip(self.sentiment_labels, prediction)
            }
            return predicted_sentiment, probabilities
        else:
            return predicted_sentiment
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
        
        Returns:
            list: List of predicted sentiments
        """
        sentiments = []
        for text in texts:
            sentiment = self.predict(text)
            sentiments.append(sentiment)
        return sentiments


def interactive_prediction():
    """Interactive mode for real-time prediction"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS - INTERACTIVE MODE")
    print("="*60)
    print("\nEnter text to analyze (or 'quit' to exit)")
    
    # Initialize predictor
    model_path = "models/saved_models/bi_lstm_sentiment_model.h5"
    tokenizer_path = "models/tokenizer.pkl"
    label_encoder_path = "models/label_encoder.pkl"
    
    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path]):
        print("\n❌ Error: Model files not found!")
        print("Please train the model first using: python src/train.py")
        return
    
    predictor = SentimentPredictor(model_path, tokenizer_path, label_encoder_path)
    
    while True:
        print("\n" + "-"*60)
        user_text = input("Enter text: ").strip()
        
        if user_text.lower() == 'quit':
            print("\nExiting... 👋")
            break
        
        if not user_text:
            print("Please enter some text!")
            continue
        
        # Predict
        sentiment, probabilities = predictor.predict(user_text, return_proba=True)
        
        # Display results
        print(f"\n📝 Input: {user_text}")
        print(f"\n🎯 Predicted Sentiment: {sentiment.upper()}")
        print("\n📊 Probabilities:")
        for label, prob in probabilities.items():
            bar = "█" * int(prob * 50)
            print(f"  {label:10s}: {prob:.4f}  {bar}")


def batch_prediction():
    """Batch prediction mode"""
    print("\n" + "="*60)
    print("BATCH SENTIMENT PREDICTION")
    print("="*60)
    
    # Sample texts
    test_texts = [
        "This course was amazing! I learned so much.",
        "Terrible experience, waste of time and money.",
        "The content was okay but could be better.",
        "Excellent instructor and great materials!",
        "Very disappointed with the quality.",
        "It was good overall."
    ]
    
    print("\nTest texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    # Initialize predictor
    model_path = "models/saved_models/bi_lstm_sentiment_model.h5"
    tokenizer_path = "models/tokenizer.pkl"
    label_encoder_path = "models/label_encoder.pkl"
    
    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path]):
        print("\n❌ Error: Model files not found!")
        print("Please train the model first using: python src/train.py")
        return
    
    predictor = SentimentPredictor(model_path, tokenizer_path, label_encoder_path)
    
    # Predict
    print("\n" + "="*60)
    print("PREDICTIONS:")
    print("="*60)
    
    for text in test_texts:
        sentiment, probs = predictor.predict(text, return_proba=True)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        for label, prob in probs.items():
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict sentiment from text')
    parser.add_argument('--text', type=str, help='Text to predict')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--batch', action='store_true', help='Batch mode with sample texts')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_prediction()
    elif args.batch:
        batch_prediction()
    elif args.text:
        # Single prediction
        model_path = "models/saved_models/bi_lstm_sentiment_model.h5"
        tokenizer_path = "models/tokenizer.pkl"
        label_encoder_path = "models/label_encoder.pkl"
        
        if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path]):
            print("\n❌ Error: Model files not found!")
            print("Please train the model first using: python src/train.py")
        else:
            predictor = SentimentPredictor(model_path, tokenizer_path, label_encoder_path)
            sentiment, probs = predictor.predict(args.text, return_proba=True)
            print(f"\nText: {args.text}")
            print(f"Predicted Sentiment: {sentiment}")
            print("\nProbabilities:")
            for label, prob in probs.items():
                print(f"  {label}: {prob:.4f}")
    else:
        print("Usage:")
        print("  python predict.py --text 'Your text here'")
        print("  python predict.py --interactive")
        print("  python predict.py --batch")