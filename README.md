# 📉 Telco Customer Churn Prediction: An Applied ML & Statistics Approach

## 🎯 Project Objective
This project was developed with a primary focus on **practicing and deeply understanding core Machine Learning algorithms and Applied Statistics concepts**. 

Instead of relying solely on high-level libraries like `scikit-learn`, the core algorithms in this project—including Logistic Regression (with L1/L2 Regularization), K-Nearest Neighbors (with Mahalanobis distance), and SMOTE—were **built from scratch using Numpy**. This approach ensures a solid mathematical foundation and a comprehensive grasp of the algorithms' inner workings.

## 📊 Dataset
* **Source:** Telco Customer Churn dataset.
* **Context:** Predicting behavior to retain customers. The dataset includes information about customer demographics, services signed up for, and account information.
* **Target Variable:** `Churn` (Yes/No).

## 🚀 Key Features & Methodologies

### 1. Exploratory Data Analysis (EDA) & Statistical Testing
* Conducted descriptive analysis on customer demographics and services.
* **Chi-Square Test of Independence:** Implemented rigorous statistical hypothesis testing ($p-value < 0.05$) to mathematically prove the correlation between categorical features (e.g., Contract type, Tech Support) and the target variable `Churn`. Feature selection was guided by statistical significance.

### 2. Data Preprocessing & Handling Imbalanced Data
* **Encoding:** Transformed categorical variables using One-Hot Encoding.
* **Scaling:** Applied Standardization to continuous variables to prevent data leakage.
* **Custom SMOTE (Synthetic Minority Over-sampling Technique):** Implemented the SMOTE algorithm from scratch using Numpy and Euclidean distance interpolation to balance the training data (addressing the 74:26 class imbalance).

### 3. Machine Learning Models (Built from Scratch)
* **Custom Logistic Regression:**
  * Implemented Gradient Descent optimization.
  * Added hyperparameter controls for **L1 (Lasso)** and **L2 (Ridge)** regularization.
* **Custom K-Nearest Neighbors (KNN):**
  * Supported multiple distance metrics: Euclidean ($p=2$), Manhattan ($p=1$).
  * **Mahalanobis Distance:** Integrated Covariance Matrix and Pseudo-inverse calculations to handle correlated features.
  * Supported both `uniform` and distance-weighted voting mechanisms.

### 4. Model Evaluation
* Evaluated models without `sklearn.metrics` by manually calculating the **Confusion Matrix**.
* Extracted key metrics: Accuracy, Precision, **Recall** (prioritized for this business problem), and F1-Score.
* **Feature Importance:** Extracted weights from the Custom Logistic Regression model to explain which factors push customers to churn.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas, Numpy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 📂 Repository Structure
* `01_EDA_and_Statistical_Analysis.ipynb`: Data exploration, Chi-square testing, and data cleaning.
* `02_Machine_Learning_Modeling.ipynb`: Data encoding, custom SMOTE, model training, and evaluation.
* `CustomLogisticRegression.py`: OOP implementation of Logistic Regression.
* `CustomKNN.py`: OOP implementation of K-Nearest Neighbors.

---
*Developed as a practical implementation of Introduction to Machine Learning and Applied Statistics methodologies.*
