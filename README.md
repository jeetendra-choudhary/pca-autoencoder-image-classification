# 🧠 Dimensionality Reduction and Representation Learning for Image Classification

> A comprehensive exploration of **Principal Component Analysis (PCA)**, **Randomized PCA**, and **Autoencoders** (linear, shallow, and deep convolutional) for image feature extraction, dimensionality reduction, and reconstruction.

---

## 📘 Project Overview

This project bridges **classical statistical learning** and **deep neural representation learning**, showing how both approaches can extract informative low-dimensional embeddings from high-dimensional image data.  

We systematically compare:
- **Standard PCA** and **Randomized PCA**
- **Linear Autoencoder (AE)**
- **Sigmoid Autoencoder (AE)**
- **Deep Convolutional Autoencoder (CAE)**

Each method is evaluated based on:
- Reconstruction fidelity  
- Classification performance  
- Eigenvector similarity  
- Computational efficiency  

---

## 🧩 Task 1 — PCA-based Dimensionality Reduction & Classification

### 🎯 Objective
Perform **Standard PCA** and **Randomized PCA** on 70% of the training dataset to extract components explaining **95% of total variance**, then train a **Logistic Regression** classifier to classify images into 10 categories.

### ⚙️ Steps
1. **Data Preparation**
   - Dataset: MNIST (10 classes)
   - Split: 70% training, 30% testing
   - Standardization: zero mean, unit variance

2. **Standard PCA**
   - Compute covariance matrix  
   - Eigen decomposition → eigenvalues & eigenvectors  
   - Retain top `k` eigenvectors explaining 95% variance  
   - Project data onto `k`-dimensional space

3. **Randomized PCA**
   - Faster approximation using randomized SVD  
   - Compare runtime & variance retention vs standard PCA

4. **Classification**
   - Train Logistic Regression on PCA features  
   - Evaluate on test set  
   - Draw ROC curves (one-vs-rest) for all classes  

### 📊 Results
| Method | Variance Retained | Accuracy | ROC-AUC | Runtime |
|--------|-------------------|-----------|----------|----------|
| Standard PCA | 95% | 92.4% | 0.97 | 24s |
| Randomized PCA | 94.8% | 92.2% | 0.97 | 5s |

### 🖼️ Visualizations
- Eigenvalue spectrum plot  
- Top 25 eigenvectors as grayscale filters  
- ROC curves per class  

---

## 🧠 Task 2 — Linear Autoencoder and PCA Comparison

### 🎯 Objective
Train a **single-layer linear autoencoder** to learn latent features equivalent to PCA components and compare visually and analytically.

### 🧩 Architecture
| Layer | Type | Activation | Shape |
|--------|------|-------------|--------|
| Input | Dense | Linear | 784 |
| Encoder | Dense | Linear | 64 |
| Decoder | Dense | Linear | 784 |

### 🧮 Training Constraints
- Encoder and decoder share weights (`W_dec = W_encᵀ`)
- Each weight vector normalized to unit magnitude
- Optimized reconstruction loss:  
  \[
  L = ||x - W(Wᵀx)||^2
  \]

### 📊 Observations
- Learned filters closely resemble PCA eigenvectors  
- Reconstruction loss converged rapidly due to orthogonal structure  
- PCA and AE subspaces aligned with cosine similarity > 0.98  

### 🖼️ Visualization
| PCA Eigenvectors | Linear AE Weights |
|------------------|-------------------|
| <img src="images/pca_filters.png" width="300"/> | <img src="images/ae_weights.png" width="300"/> |

### 💬 Comment
Both PCA and Linear Autoencoder capture global variance directions. The AE representation, however, may differ slightly due to stochastic gradient optimization.

---

## 🧩 Task 3 — Deep Convolutional Autoencoder (CAE) vs Shallow Autoencoders

### 🎯 Objective
Design a **Deep Convolutional Autoencoder (CAE)** with the same latent dimension as Task 2 and compare its reconstruction accuracy with:
- Single-layer Sigmoid Autoencoder  
- Hypothetical 3-layer Autoencoder  

### ⚙️ Architectures

#### 1️⃣ Single-layer Sigmoid Autoencoder
```python
Encoder: Dense(64, activation='sigmoid')
Decoder: Dense(784, activation='linear')
