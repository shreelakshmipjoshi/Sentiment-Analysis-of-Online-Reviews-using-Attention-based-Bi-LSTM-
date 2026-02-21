# Sentiment Analysis of Online Course Reviews

## Attention-Based Bi-LSTM Neural Network

A deep learning project that analyzes student course reviews using Bidirectional LSTM with attention mechanism for accurate sentiment classification.

---

## 📋 Project Overview

This project implements an advanced Natural Language Processing (NLP) solution to analyze and classify sentiment in online course reviews. The model uses Bidirectional Long Short-Term Memory (Bi-LSTM) networks combined with an attention mechanism to achieve high accuracy in sentiment classification.

### Key Features
- ✅ Bi-LSTM architecture for bidirectional context processing
- ✅ Attention mechanism for focusing on important words and phrases
- ✅ Multi-class sentiment classification (Positive, Negative, Neutral)
- ✅ Comprehensive data preprocessing pipeline
- ✅ Model evaluation with multiple metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Detailed technical documentation and analysis

---

## 🛠️ Technology Stack

### Core Technologies
- **Python** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities
- **NLTK** - Natural language toolkit

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git/GitHub** - Version control
- **VS Code** - Code editor

---

## 📁 Project Structure

```
sentiment-analysis-bi-lstm/
│
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/              # Preprocessed data
│   └── train_test_split/       # Training and testing splits
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preprocessing
│   ├── model.py                # Bi-LSTM model definition
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── attention.py            # Attention mechanism implementation
│
├── models/
│   └── saved_models/           # Trained model checkpoints
│
├── reports/
│   ├── Sentiment Analysis of Online Course Reviews Using Attention-Based Bi-LSTM.docx
│   └── evaluation_metrics.txt
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                 # Git ignore file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-bi-lstm.git
   cd sentiment-analysis-bi-lstm
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset

### Dataset Description
- **Source**: Online course reviews from educational platforms
- **Format**: CSV/JSON with review text and sentiment labels
- **Classes**: 
  - Positive (1)
  - Negative (0)
  - Neutral (2)

### Data Preprocessing Steps
1. Text cleaning (removing special characters, URLs, etc.)
2. Tokenization using NLTK
3. Stop word removal
4. Padding sequences to fixed length
5. Word embedding (GloVe/Word2Vec)
6. Train-test split (80-20 ratio)

---

## 🧠 Model Architecture

### Bi-LSTM with Attention

```
Input Layer (Tokenized Text)
    ↓
Embedding Layer (Pre-trained Word Vectors)
    ↓
Bi-LSTM Layer (Forward + Backward)
    ↓
Attention Mechanism
    ↓
Dense Layer (ReLU Activation)
    ↓
Output Layer (Softmax - 3 Classes)
```

### Model Specifications
- **Embedding Dimension**: 100-300
- **LSTM Units**: 128-256 (bidirectional)
- **Attention Heads**: Multi-head attention
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 📈 Model Performance

### Evaluation Metrics

| Metric | Score |
|--------|-------|
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1-Score | XX% |

*(Update with actual model performance after training)*

### Confusion Matrix & Classification Report
Detailed analysis included in project report.

---

## 🏃 Usage

### Training the Model

```bash
python src/train.py --data data/processed/train.csv --epochs 50 --batch_size 32
```

### Evaluating the Model

```bash
python src/evaluate.py --model models/saved_models/bi_lstm_attention.h5
```

### Making Predictions

```python
from src.model import load_model, predict_sentiment

model = load_model('models/saved_models/bi_lstm_attention.h5')
sentiment = predict_sentiment(model, "This course was amazing!")
print(f"Predicted Sentiment: {sentiment}")
```

---

## 📈 Results & Analysis

### Key Findings
- Bi-LSTM outperforms unidirectional LSTM by capturing context from both directions
- Attention mechanism significantly improves model interpretability
- Model performs well on identifying strongly positive/negative reviews
- Neutral sentiment classification remains challenging

### Visualization
- Training/validation loss curves
- Accuracy over epochs
- Confusion matrix
- Attention weight visualization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Project Reports

- **Technical Report**: `reports/Sentiment Analysis of Online Course Reviews Using Attention-Based Bi-LSTM.docx`
- Complete documentation available in project folder

---

## 📓 Google Colab Notebook

**Interactive Jupyter Notebook**: [Open in Colab](https://colab.research.google.com/drive/15e2ib7OoflP0eSjcN0nYN5cQqFDDik4S?usp=sharing)

- Run the complete project in your browser
- No local setup required
- Includes all code cells for training and prediction

---

## 🎯 Future Improvements

- [ ] Implement transformer-based models (BERT, RoBERTa)
- [ ] Add support for multiple languages
- [ ] Deploy as a web application
- [ ] Real-time sentiment analysis API
- [ ] Aspect-based sentiment analysis
- [ ] Integrate with course platforms for live feedback

---

## 📧 Contact

**Shreelakshmi P. Joshi**
- Email: shreelakshmipjoshi@gmail.com
- LinkedIn: https://www.linkedin.com/in/shreelakshmi-p-joshi-564a10266
- Portfolio: https://shreelakshmi-joshi-portfolio.netlify.app

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---



## ⭐ Star this Project!

If you find this project helpful, please consider giving it a star! ⭐

---

**Last Updated**: February 2026
**Project Status**: ✅ Complete
**Python Version**: 3.8+
