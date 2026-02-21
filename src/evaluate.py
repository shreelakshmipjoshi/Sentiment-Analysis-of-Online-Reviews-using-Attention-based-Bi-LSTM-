"""
Evaluation Script for Sentiment Analysis Model

This script provides detailed evaluation metrics and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data, prepare_for_training


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    class_names = label_encoder.classes_.tolist()
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print(cm)
    
    # Per-class metrics
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    for class_name in class_names:
        metrics = report[class_name]
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }


def plot_confusion_matrix(cm, class_names, save_path='models/confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_training_history(history_path='models/training_history.csv', save_path='models/training_curves.png'):
    """
    Plot training curves
    
    Args:
        history_path: Path to training history CSV
        save_path: Path to save the plot
    """
    if not os.path.exists(history_path):
        print(f"\nTraining history not found at: {history_path}")
        return
    
    history_df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def save_evaluation_report(metrics, save_path='models/evaluation_report.txt'):
    """
    Save evaluation report to text file
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SENTIMENT ANALYSIS MODEL - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Loss: {metrics['loss']:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n")
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict):
                f.write(f"\n{class_name.upper()}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                if 'support' in class_metrics:
                    f.write(f"  Support: {class_metrics['support']}\n")
    
    print(f"\nEvaluation report saved to: {save_path}")


if __name__ == "__main__":
    print("\nLoading and evaluating model...")
    
    # Paths
    model_path = "models/saved_models/bi_lstm_sentiment_model.h5"
    data_path = "data/raw/updated_sentiment_analysis_dataset.csv"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train the model first using: python src/train.py")
        sys.exit(1)
    
    # Load model
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Load and prepare data
    data = load_and_preprocess_data(data_path)
    X_test, y_test, tokenizer, label_encoder = prepare_for_training(data)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(metrics['confusion_matrix'], label_encoder.classes_)
    plot_training_history()
    
    # Save report
    save_evaluation_report(metrics)
    
    print("\n✅ Evaluation completed!")
    print("\nGenerated files:")
    print("  - models/confusion_matrix.png")
    print("  - models/training_curves.png")
    print("  - models/evaluation_report.txt")