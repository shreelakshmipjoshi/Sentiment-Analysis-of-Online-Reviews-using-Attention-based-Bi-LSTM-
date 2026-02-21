"""
Training Script for Sentiment Analysis Model

This script trains the Bi-LSTM sentiment analysis model
"""

import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data, prepare_for_training
from model import build_bilstm_model, print_model_summary


def train_model(
    data_filepath,
    model_save_path='models/sentiment_model.h5',
    epochs=10,
    batch_size=64,
    max_vocab_size=10000,
    max_seq_len=100,
    embedding_dim=128,
    lstm_units=128
):
    """
    Train the sentiment analysis model
    
    Args:
        data_filepath (str): Path to the CSV dataset
        model_save_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        max_vocab_size (int): Maximum vocabulary size
        max_seq_len (int): Maximum sequence length
        embedding_dim (int): Embedding dimension
        lstm_units (int): Number of LSTM units
    
    Returns:
        tuple: (model, history, tokenizer, label_encoder)
    """
    print("="*80)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*80)
    
    # Load and preprocess data
    print("\n[Step 1/4] Loading and preprocessing data...")
    data = load_and_preprocess_data(data_filepath)
    
    # Prepare for training
    print("\n[Step 2/4] Preparing data for training...")
    X_train, X_test, y_train, y_test, tokenizer, label_encoder = prepare_for_training(
        data, 
        max_vocab_size=max_vocab_size, 
        max_seq_len=max_seq_len
    )
    
    # Build model
    print("\n[Step 3/4] Building model architecture...")
    model = build_bilstm_model(
        max_vocab_size=max_vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        lstm_units=lstm_units,
        num_classes=3
    )
    print_model_summary(model)
    
    # Create models directory if it doesn't exist
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[Step 4/4] Training model...")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Validation Samples: {X_test.shape[0]}")
    print("\nStarting training...\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'train_accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy']
    })
    history_path = 'models/training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    return model, history, tokenizer, label_encoder


if __name__ == "__main__":
    # Training parameters
    DATA_PATH = "data/raw/updated_sentiment_analysis_dataset.csv"
    MODEL_PATH = "models/saved_models/bi_lstm_sentiment_model.h5"
    
    # Train the model
    model, history, tokenizer, label_encoder = train_model(
        data_filepath=DATA_PATH,
        model_save_path=MODEL_PATH,
        epochs=10,
        batch_size=64,
        max_vocab_size=10000,
        max_seq_len=100,
        embedding_dim=128,
        lstm_units=128
    )
    
    print("\n✅ Training completed successfully!")
    print("\nNext steps:")
    print("1. Use 'predict.py' to make predictions on new text")
    print("2. Use 'evaluate.py' for detailed evaluation metrics")