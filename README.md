# Deep Learning Laboratory

A comprehensive repository containing laboratory assignments, implementations, datasets, and experiments for the **Deep Learning** course (VIT B.Tech). The lab covers fundamental and advanced deep learning concepts ranging from multi-layer perceptrons to recurrent neural networks, convolutional neural networks, transfer learning, transformer-based NLP models, and cutting-edge Vision Transformers (ViT).

---

## 📚 Table of Contents

- [Overview of Assignments](#-overview-of-assignments)
- [Detailed Assignment Breakdown](#-detailed-assignment-breakdown)
  - [Assignment 1: TensorFlow & Keras Foundations (MNIST)](#assignment-1-tensorflow--keras-foundations-mnist)
  - [Assignment 2: Multi-Layer Perceptron (Iris Classification)](#assignment-2-multi-layer-perceptron-iris-classification)
  - [Assignment 3: Forward/Backpropagation & Hyperparameter Tuning](#assignment-3-forwardbackpropagation--hyperparameter-tuning)
  - [Assignment 4: Time-Series Forecasting with LSTM (Airline Passengers)](#assignment-4-time-series-forecasting-with-lstm-airline-passengers)
  - [Assignment 5: Sequence Modeling with RNN, LSTM & GRU (IMDb Sentiment)](#assignment-5-sequence-modeling-with-rnn-lstm--gru-imdb-sentiment)
  - [Assignment 6: CNN for Tomato Leaf Disease Classification](#assignment-6-cnn-for-tomato-leaf-disease-classification)
  - [Assignment 7: Transfer Learning with Pre-trained CNNs (CIFAR-10)](#assignment-7-transfer-learning-with-pre-trained-cnns-cifar-10)
  - [Assignment 8: Fine-Tuning BERT for Text Classification (AG News)](#assignment-8-fine-tuning-bert-for-text-classification-ag-news)
  - [Assignment 9: Vision Transformer (ViT) vs. CNN (ResNet-18)](#assignment-9-vision-transformer-vit-vs-cnn-resnet-18)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
- [Data & Model Checkpoint Management](#-data--model-checkpoint-management)

---

## 🔬 Overview of Assignments

| # | Topic | Architecture / Technique | Dataset | Framework |
|---|-------|-------------------------|---------|-----------|
| **1** | Deep Learning Framework Basics | Sequential MLP | MNIST Handwritten Digits | TensorFlow / Keras |
| **2** | Multiclass Classification | Multi-Layer Perceptron (MLP) | Iris Dataset | TensorFlow / Keras |
| **3** | Optimization & Learning Dynamics | Forward & Backpropagation, LR Tuning | Iris Dataset | TensorFlow / Keras |
| **4** | Time-Series Forecasting | LSTM (Windowing: 6, 12, 24 months) | Airline Passengers | TensorFlow / Keras |
| **5** | Natural Language & Sequence Modeling | Simple RNN vs. LSTM vs. GRU | IMDb Movie Reviews | TensorFlow / Keras |
| **6** | Computer Vision & Image Classification | Deep Convolutional Neural Network (CNN) | Tomato Leaf Disease (10K+ images) | TensorFlow / Keras |
| **7** | Transfer Learning & Model Comparison | AlexNet, VGG-16, ResNet-50, EfficientNet-B0 | CIFAR-10 | PyTorch / Torchvision |
| **8** | Transformers & NLP Fine-Tuning | Pre-trained BERT (`bert-base-uncased`) | AG News (Multiclass) | PyTorch / Hugging Face Transformers |
| **9** | Vision Transformers vs. CNNs | ViT-B/16 vs. ResNet-18 (Head-to-Head Benchmark) | CIFAR-10 | PyTorch / Torchvision |

---

## 📖 Detailed Assignment Breakdown

### Assignment 1: TensorFlow & Keras Foundations (MNIST)
- **Goal:** Introduction to TensorFlow and Keras APIs.
- **Workflow:**
  - Loading and normalizing the MNIST handwritten digits dataset.
  - Building a feedforward neural network with Dense, Dropout, and Softmax activation.
  - Training, evaluating on test split, and exporting the trained model (`my_mnist_model.keras`).

### Assignment 2: Multi-Layer Perceptron (Iris Classification)
- **Goal:** Constructing an MLP for multiclass classification on tabular features.
- **Workflow:**
  - Data preprocessing using `StandardScaler` and stratified train-test splitting.
  - Designing a multi-layer feedforward architecture with ReLU and Softmax.
  - Evaluating performance with accuracy metrics and confusion matrices.

### Assignment 3: Forward/Backpropagation & Hyperparameter Tuning
- **Goal:** Analyzing convergence, gradient descent dynamics, and loss landscapes.
- **Workflow:**
  - Experimenting with varying learning rates ($\alpha$) and epoch counts.
  - Visualizing loss curves and validation metrics to observe underfitting, optimal convergence, and oscillations.

### Assignment 4: Time-Series Forecasting with LSTM (Airline Passengers)
- **Goal:** Sequence-to-point time-series prediction using Recurrent architectures.
- **Workflow:**
  - Feature normalization with `MinMaxScaler`.
  - Look-back window generation for historical context: **6-month**, **12-month**, and **24-month** windows.
  - Comparative evaluation using RMSE and MAE to determine the optimal historical context for long-term forecasting.

### Assignment 5: Sequence Modeling with RNN, LSTM & GRU (IMDb Sentiment)
- **Goal:** Comparative study of recurrent neural network variants on text sentiment classification.
- **Workflow:**
  - Text tokenization and sequence padding to fixed sequence lengths.
  - Embedding layer integration followed by:
    1. **Simple RNN** (`rnn_model.keras`)
    2. **LSTM** (`lstm_model.keras`)
    3. **GRU** (`gru_model.keras`)
  - Benchmarking training speed, memory footprint, vanishing gradient resistance, and test accuracy across all three architectures.

### Assignment 6: CNN for Tomato Leaf Disease Classification
- **Goal:** End-to-end computer vision pipeline for agricultural disease detection.
- **Workflow:**
  - Dataset of 10,000+ tomato leaf images across multiple healthy and pathological classes.
  - Image preprocessing: resizing, rescaling ($1/255$), and batching.
  - Building a convolutional architecture with `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
  - Saved model export (`tomato_disease_cnn.keras`) with high classification accuracy.

### Assignment 7: Transfer Learning with Pre-trained CNNs (CIFAR-10)
- **Goal:** Comparing transfer learning efficacy across diverse pre-trained vision backbones.
- **Workflow:**
  - CIFAR-10 dataset loaded and normalized using ImageNet statistics.
  - Freezing pre-trained feature extraction backbones and replacing classification heads for:
    - **AlexNet**
    - **VGG-16**
    - **ResNet-50**
    - **EfficientNet-B0**
  - Standardized training loop, loss tracking, and head-to-head comparison of accuracy and parameter efficiency.

### Assignment 8: Fine-Tuning BERT for Text Classification (AG News)
- **Goal:** Applying modern Transformer architectures to multiclass text classification.
- **Workflow:**
  - Loading and exploring the AG News dataset (World, Sports, Business, Sci/Tech).
  - Subword tokenization using `BertTokenizerFast`.
  - Fine-tuning `bert-base-uncased` with Hugging Face `Trainer` API.
  - Comprehensive evaluation: Accuracy, Precision, Recall, Macro/Weighted F1-score, and Confusion Matrix.
  - Exporting tokenizer and configuration files for inference.

### Assignment 9: Vision Transformer (ViT) vs. CNN (ResNet-18)
- **Goal:** Benchmarking modern attention-based Vision Transformers against classic spatial-inductive Convolutional Networks.
- **Workflow:**
  - Deconstructing and visualizing the ViT tokenization mechanism ($16 \times 16$ image patch decomposition yielding $196$ visual tokens).
  - Comparing **ResNet-18** (11.2M parameters) with **ViT-B/16** (85.8M parameters) under identical transfer learning setups on CIFAR-10.
  - Rigorous benchmarking: Loss/Accuracy learning curves, Macro & Weighted F1-scores, Confusion Matrices, and Inference Latency per sample.
  - In-depth theoretical analysis of inductive biases (translation equivariance & locality vs. global all-to-all self-attention).

---

## 📁 Repository Structure

```text
Deep-Learning-Lab/
│
├── Assignment 1/
│   ├── Assignment_1.ipynb
│   └── my_mnist_model.keras
│
├── Assignment 2/
│   └── Assignment_2.ipynb
│
├── Assignment 3/
│   └── Assignment_3.ipynb
│
├── Assignment 4/
│   └── Assignment_4.ipynb
│
├── Assignment 5/
│   ├── Assignment_5.ipynb
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   └── rnn_model.keras
│
├── Assignment 6/
│   └── tomato/
│       ├── cnn_train.ipynb
│       ├── tomato_disease_cnn.keras
│       ├── train/                      # Tomato dataset train split
│       └── val/                        # Tomato dataset validation split
│
├── Assignment 7/
│   ├── Assignment_7.ipynb
│   └── data/
│       └── cifar-10-batches-py/        # Extracted CIFAR-10 dataset batches
│
├── Assignment 8/
│   ├── Assignment_8.ipynb
│   └── bert_agnews_model/              # Fine-tuned BERT model configs & tokenizer
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── Assignment 9/
│   └── Assignment_9.ipynb              # Vision Transformer (ViT-B/16) vs. CNN (ResNet-18)
│
├── .gitignore                          # Clean ignore rules for large weights & envs
└── README.md
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Manthan270706/Deep-Learning-Lab.git
cd Deep-Learning-Lab
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv dl_env
dl_env\Scripts\activate

# Linux / macOS
python3 -m venv dl_env
source dl_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install tensorflow torch torchvision transformers datasets scikit-learn pandas numpy matplotlib seaborn jupyter
```

---

## 📦 Data & Model Checkpoint Management

- **Datasets:**
  - The tomato leaf disease dataset (`Assignment 6`) and CIFAR-10 batch files (`Assignment 7`) are directly included in the repository.
- **Model Checkpoints:**
  - Lightweight `.keras` models are tracked in the repository.
  - Massive PyTorch weight checkpoints (`*.pth`), intermediate optimizer states (`*.pt`), raw `.tar.gz` archives, and large safetensors files (>100 MB) are excluded via `.gitignore` to maintain a clean repository adhering to GitHub file size limits.

---

## 👤 Author
- **Manthan Sali**
- Repository: [Deep-Learning-Lab](https://github.com/Manthan270706/Deep-Learning-Lab)
