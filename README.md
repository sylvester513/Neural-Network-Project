# 🩺 Breast Cancer Prediction using Machine Learning & Neural Networks

A supervised machine learning project that predicts whether a breast tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** using the **Wisconsin Breast Cancer Dataset** loaded from `sklearn.datasets.load_breast_cancer`.

The project covers the full ML pipeline: **data loading → cleaning → EDA → feature scaling → model training → Neural Network → evaluation → prediction**.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers among women worldwide. Early and accurate detection is critical for successful treatment. This project builds an intelligent diagnostic system that:
- Loads the built-in breast cancer dataset from Scikit-learn
- Cleans and preprocesses the data
- Trains multiple supervised ML models
- Builds a Neural Network for higher accuracy
- Compares model performance
- Predicts whether a tumor is malignant or benign

---

## 🎯 Objectives

- Load and explore the `load_breast_cancer` dataset
- Perform data cleaning and preprocessing
- Handle feature scaling and class distribution
- Train and evaluate multiple classification models
- Build and tune a Neural Network
- Deploy the best model for real-time prediction

---

## 🗂️ Dataset

**Source:** `sklearn.datasets.load_breast_cancer`

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 numeric features |
| Classes | 2 (Malignant = 0, Benign = 1) |
| Missing Values | None |
| Task | Binary Classification |

**Features (31 total):**

| Category | Features |
|----------|----------|
| **Mean** | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| **Standard Error** | same 10 features (SE) |
| **Worst** | same 10 features (worst) |

**Target:**
- `0` → Malignant (cancerous)
- `1` → Benign (non-cancerous)

---

## 🧹 Data Cleaning & Preprocessing

Steps performed:
- Loaded dataset using `load_breast_cancer()` from `sklearn.datasets`
- Converted to Pandas DataFrame
- Checked for missing values (none found)
- Checked for duplicates and removed them
- Detected and handled outliers using IQR
- Scaled features using **StandardScaler**
- Split data into **80% train / 20% test** (stratified)
- Applied **SMOTE** (optional) for class balance

---

## 🔍 Exploratory Data Analysis (EDA)

- Class distribution plot (Malignant vs Benign)
- Correlation heatmap of 30 features
- Distribution plots for key features (`mean radius`, `mean area`, `mean concavity`)
- Boxplots for outlier detection
- Pairplot of top discriminative features

---

## 🤖 Models Trained

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline |
| Decision Tree | Interpretability |
| Random Forest | Ensemble accuracy |
| XGBoost / LightGBM | High performance |
| Support Vector Machine (SVM) | Margin-based |
| K-Nearest Neighbors (KNN) | Distance-based |
| **Neural Network (MLP)** | Deep learning approach |

---


## 🧠 Neural Network Architecture

Built using **TensorFlow / Keras**:
