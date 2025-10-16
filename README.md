# TMNIST Alphabet (94 Characters) - Neural Network Classifier

A machine learning project that classifies handwritten characters and fonts using a 2-layer neural network. This model recognizes 94 character classes including uppercase letters (A-Z), lowercase letters (a-z), digits (0-9), and punctuation marks.

## Overview

This project implements a neural network from scratch using NumPy to classify characters from the TMNIST (Typed Modified NIST) dataset. The model achieves approximately **84% accuracy** on the test set.

**Dataset:** 94 character classes with 28×28 pixel images  
**Model Type:** 2-layer Feedforward Neural Network  
**Activation Functions:** ReLU (hidden layer) + Softmax (output layer)

## Dataset

The TMNIST Alphabet dataset contains:
- **94 character classes:** A-Z, a-z, 0-9, and punctuation (!, ", /, etc.)
- **28×28 pixels** per image (784 features)
- **3000+ samples** across different fonts and styles
- **Features:** Normalized pixel values (0-1 range)

### Sample Characters Recognized
```
Uppercase:  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
Lowercase:  a b c d e f g h i j k l m n o p q r s t u v w x y z
Digits:     0 1 2 3 4 5 6  8 9
Symbols:    ! " # $ % & ' ( ) * + , - . / : ; < = > ? [ \ ] ^ _ ` { | } ~
```

## Model Architecture
```
Input Layer (84 features)
    ↓
Hidden Layer (512 neurons, ReLU activation)
    ↓
Output Layer (94 neurons, Softmax activation)
```

### Network Details
- **Weights:** Randomly initialized from range [-0.5, 0.5]
- **Activation:** ReLU for hidden layer, Softmax for output
- **Loss Function:** Categorical Cross-Entropy (via Softmax)
- **Optimizer:** Gradient Descent
- **Learning Rate:** 0.1
- **Training Iterations:** 1000

## Performance

| Metric | Accuracy |
|--------|----------|
| Training Accuracy | 85.2% |
| Development Accuracy | 84.05% |
| Test Accuracy | 83.4% |

The model stabilizes around 82-83% accuracy after ~500 iterations.

## Installation & Setup

### Requirements
- Python 3.9+
- Conda (recommended)
- Dataset-https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters

## Usage

### Running the Complete Training Pipeline
```python
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('tmnist.csv')

# Prepare features and labels
labels = df['labels'].values
pixels = df.iloc[:, 2:86].values

# Normalize pixel values
X = pixels / 255.0

# Initialize weights
W1 = np.random.rand(512, 84) - 0.5   # Hidden: 512 neurons
b1 = np.random.rand(512, 1) - 0.5
W2 = np.random.rand(94, 512) - 0.5    # Output: 94 classes
b2 = np.random.rand(94, 1) - 0.5

# Train model (1000 iterations)
# Forward pass → Backpropagation → Weight updates
```

### Making Predictions
```python
def predict(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = np.maximum(Z1, 0)  # ReLU
    Z2 = W2.dot(A1) + b2
    expZ = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
    A2 = expZ / np.sum(expZ, axis=0, keepdims=True)  # Softmax
    return np.argmax(A2, axis=0)

# Get predictions
predictions = predict(X_test, W1, b1, W2, b2)
```

### Visualizing Results

The notebook includes visualization of:
- Sample character images from the dataset
- Model predictions vs actual labels
- Training accuracy progression
- Character class distribution

## Key Functions

### Forward Propagation
```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

### Backward Propagation
```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_derive(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    
    return dW1, db1, dW2, db2
```


## Training Process Visualization

**Accuracy progression during training:**
```
Iteration 0 - Accuracy: 1.9%
Iteration 10 - Accuracy: 6.65%
Iteration 20 - Accuracy: 1.44%
Iteration 30 - Accuracy: 30.81%
Iteration 40 - Accuracy: 42.81%
Iteration 50 - Accuracy: 51.80%
Iteration 60 - Accuracy: 58.56%
Iteration 0 - Accuracy: 63.40%
Iteration 80 - Accuracy: 66.88%
....
Iteration 80 - Accuracy: 83.32%
Iteration 880 - Accuracy: 83.36%
Iteration 890 - Accuracy: 83.40%
Iteration 900 - Accuracy: 83.43%
Iteration 910 - Accuracy: 83.4%✓
```

The model converges around iteration 500-600, with diminishing improvements afterward.

## Implementation Details

### Data Preprocessing
- Images resized to 28×28 pixels (84 features)
- Pixel values normalized to [0, 1] range
- Labels mapped to numeric indices (0-93)
- Data split: 0% training, 30% validation/test

### Hyperparameters
- **Hidden Units:** 512
- **Learning Rate (α):** 0.1
- **Iterations:** 1000
- **Weight Initialization:** Uniform random [-0.5, 0.5]

### Loss Function
Cross-entropy loss with Softmax:
```
L = -Σ(y * log(ŷ))
```


