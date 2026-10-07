# 🩺 Breast Cancer Prediction Using Machine Learning & Neural Networks

A supervised machine learning project that predicts whether a breast tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** using the **Wisconsin Breast Cancer Dataset** from `sklearn.datasets.load_breast_cancer`.

The project covers the complete machine learning process, from loading and preparing the dataset to training different classification models, building a Neural Network, evaluating the models, and making predictions.

---

## 📌 Project Overview

Breast cancer detection is an important classification problem where machine learning can be used to identify patterns in tumor measurements and assist in distinguishing between malignant and benign tumors.

In this project, I worked with the breast cancer dataset provided by Scikit-learn and followed a complete machine learning workflow.

The project includes:

* Loading the breast cancer dataset
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature scaling
* Training different supervised machine learning models
* Comparing model performance
* Building a Neural Network using **TensorFlow and Keras**
* Monitoring training and validation performance
* Evaluating the Neural Network using `model.evaluate()`
* Making predictions for malignant and benign tumors

---

## 🎯 Objectives

* Load and explore the `load_breast_cancer` dataset
* Perform data cleaning and preprocessing
* Handle feature scaling and class distribution
* Train and evaluate multiple classification models
* Build and tune a Neural Network
* Monitor training and validation performance
* Evaluate the trained Neural Network using loss and accuracy
* Deploy the best-performing model for real-time prediction

---

## 🗂️ Dataset

**Source:** `sklearn.datasets.load_breast_cancer`

| Property           | Value                         |
| ------------------ | ----------------------------- |
| **Samples**        | 569                           |
| **Features**       | 30 numeric features           |
| **Classes**        | 2 (Malignant = 0, Benign = 1) |
| **Missing Values** | None                          |
| **Task**           | Binary Classification         |

### Features

The dataset contains measurements grouped into three main categories:

| Category           | Features                                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Mean**           | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| **Standard Error** | same 10 features (SE)                                                                                             |
| **Worst**          | same 10 features (worst)                                                                                          |

These measurements describe different characteristics of the cell nuclei present in the breast tissue samples.

### Target

* `0` → **Malignant (cancerous)**
* `1` → **Benign (non-cancerous)**

---

## 🧹 Data Cleaning & Preprocessing

The following steps were performed before training the models:

* Loaded the dataset using `load_breast_cancer()` from `sklearn.datasets`
* Converted the dataset into a Pandas DataFrame
* Checked for missing values
* Checked for duplicate records and removed them
* Detected and handled outliers using the IQR method
* Separated the input features from the target variable
* Scaled the features using **StandardScaler**
* Split the dataset into **80% training and 20% testing**
* Used stratification when splitting the dataset
* Applied **SMOTE** where required for class balancing

Feature scaling was particularly important for models such as SVM, KNN, and the Neural Network because the dataset contains features with different numerical ranges.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was used to understand the structure of the dataset and the relationships between the different features.

The analysis included:

* Class distribution plot for Malignant and Benign tumors
* Correlation heatmap of the 30 features
* Distribution plots for important features
* Distribution of `mean radius`
* Distribution of `mean area`
* Distribution of `mean concavity`
* Boxplots for detecting outliers
* Pairplot of selected discriminative features

The correlation analysis also helped identify features that have strong relationships with each other and with the target classification problem.

---

## 🤖 Models Trained

Different classification algorithms were trained and evaluated to compare their performance.

| Model                            | Purpose                                        |
| -------------------------------- | ---------------------------------------------- |
| **Logistic Regression**          | Baseline classification model                  |
| **Decision Tree**                | Interpretability and rule-based classification |
| **Random Forest**                | Ensemble-based classification                  |
| **XGBoost / LightGBM**           | High-performance gradient boosting             |
| **Support Vector Machine (SVM)** | Margin-based classification                    |
| **K-Nearest Neighbors (KNN)**    | Distance-based classification                  |
| **Neural Network (MLP)**         | Neural network approach                        |

The models were evaluated to determine which approach performed best on the breast cancer classification task.

---

# 🧠 Neural Network Using TensorFlow & Keras

One of the main parts of the project was building a Neural Network using **TensorFlow and Keras**.

Instead of only relying on traditional machine learning algorithms, I used a neural network to learn patterns from the scaled breast cancer features and classify the tumors into the two target classes.

### TensorFlow

**TensorFlow** was used as the machine learning framework responsible for the underlying numerical computations and training process.

### Keras

**Keras** was used to build the Neural Network because it provides a simpler interface for defining layers, compiling the model, training it, and evaluating its performance.

The general workflow used for the Neural Network was:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Neural Network
   ↓
Model Compilation
   ↓
Model Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Prediction
```

---

## 🏗️ Neural Network Training

The Neural Network was trained using the prepared training data.

During training, the model learned the relationship between the input features and the target variable over multiple epochs.

The training process monitored both:

* **Accuracy**
* **Loss**

The validation data was also monitored during training so that I could compare how the model performed on data that was not directly used to update the model's weights.

---

## 📊 Training & Validation Performance

During the Neural Network training process, I plotted the training and validation metrics to understand how the model was learning.

The plots included:

### Accuracy

The accuracy plot shows how the model's classification accuracy changed during training.

It compares:

* `accuracy`
* `val_accuracy`

This makes it possible to see how the model performed on the training data compared with the validation data.

### Loss

The loss plot shows how the model's error changed during training.

It compares:

* `loss`
* `val_loss`

Monitoring both training and validation loss is useful for identifying whether the model is learning properly or starting to overfit.

The training history was used to generate these plots and visualize the model's performance across the training epochs.

---

## 📈 Model Evaluation

After training the Neural Network, the model was evaluated using the test data.

I used Keras' `model.evaluate()` method to obtain the final evaluation metrics.

For example:

```python
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

This provided the final **loss** and **accuracy** of the trained Neural Network on the test dataset.

The evaluation is important because training accuracy alone does not show how well the model performs on previously unseen data.

---

## 🔎 Prediction

After training and evaluating the model, the Neural Network can be used to make predictions on new tumor measurements.

The model produces a prediction corresponding to one of the two classes:

```text
0 → Malignant
1 → Benign
```

This allows the trained model to be integrated into a prediction application where a user can provide the required tumor measurements and receive a classification result.

---

## 🛠️ Technologies & Libraries

The project uses the following technologies and Python libraries:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **TensorFlow**
* **Keras**
* **SMOTE / Imbalanced-learn**
* **Jupyter Notebook**

---

## 📚 Machine Learning Concepts Covered

This project provided practical experience with:

* Supervised Learning
* Binary Classification
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Feature Scaling
* Train/Test Splitting
* Stratified Data Splitting
* Handling Class Imbalance
* SMOTE
* Outlier Detection
* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost / LightGBM
* Support Vector Machines
* K-Nearest Neighbors
* Neural Networks
* TensorFlow
* Keras
* Model Training
* Validation
* Model Evaluation
* Model Prediction
* Training and Validation Curves

---

## ⚠️ Disclaimer

This project is intended for **educational and machine learning demonstration purposes**. The predictions generated by the model should not be considered a medical diagnosis or a replacement for professional medical advice.
