"""
Model Architecture for Sentiment Analysis

This module defines the Bi-LSTM model architecture for sentiment classification
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def build_bilstm_model(max_vocab_size=10000, embedding_dim=128, max_seq_len=100, 
                       lstm_units=128, num_classes=3, dropout_rate=0.5):
    """
    Build a Bidirectional LSTM model for sentiment classification
    
    Args:
        max_vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        max_seq_len (int): Maximum sequence length
        lstm_units (int): Number of LSTM units
        num_classes (int): Number of output classes (3 for Positive, Negative, Neutral)
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        keras.Model: Compiled Bi-LSTM model
    """
    # Input layer
    input_layer = Input(shape=(max_seq_len,), name='input_layer')
    
    # Embedding layer
    embedding_layer = Embedding(
        input_dim=max_vocab_size, 
        output_dim=embedding_dim, 
        input_length=max_seq_len,
        name='embedding_layer'
    )(input_layer)
    
    # Dropout after embedding
    x = Dropout(dropout_rate)(embedding_layer)
    
    # Bidirectional LSTM layer
    bilstm_layer = Bidirectional(
        LSTM(lstm_units, return_sequences=False, name='lstm_layer'),
        name='bidirectional_lstm'
    )(x)
    
    # Dropout after LSTM
    x = Dropout(dropout_rate)(bilstm_layer)
    
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(
        num_classes, 
        activation='softmax', 
        name='output_layer'
    )(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer, name='BiLSTM_Sentiment_Model')
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """
    Print model summary
    
    Args:
        model (keras.Model): Keras model
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80)


def save_model_architecture(model, filepath='model_architecture.png'):
    """
    Save model architecture diagram
    
    Args:
        model (keras.Model): Keras model
        filepath (str): Path to save the diagram
    """
    try:
        plot_model(model, to_file=filepath, show_shapes=True, show_layer_names=True)
        print(f"\nModel architecture saved to: {filepath}")
    except Exception as e:
        print(f"\nCould not save model architecture: {e}")


if __name__ == "__main__":
    # Build and display model
    print("Building Bi-LSTM Model...")
    model = build_bilstm_model()
    print_model_summary(model)