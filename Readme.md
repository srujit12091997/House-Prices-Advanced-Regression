# 🏡 House Prices - Advanced Regression Techniques

Predicting house prices using advanced regression techniques to assist buyers, sellers, and real estate analysts in making informed decisions.

![Kaggle Badge](https://img.shields.io/badge/Kaggle-House%20Prices-blue)  
[Competition Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📑 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [How to Run](#how-to-run)
---

## 🧠 Problem Statement

The goal is to predict final house sale prices based on various features like square footage, location, condition, and more using regression models. This is a supervised machine learning problem focused on **feature engineering**, **handling missing data**, and **model stacking**.

---

## 🗃 Dataset

The dataset provided by Kaggle contains:
- **1460 training examples** with 81 features (numeric and categorical)
- **Test set with similar features but without sale price**

Key features include:
- Lot Area
- Neighborhood
- Overall Condition
- Year Built
- Garage Area

---

## 🔍 Approach

1. **Data Preprocessing**
   - Handling missing values
   - Outlier removal
   - Label encoding for categorical data

2. **Feature Engineering**
   - Log-transform of skewed features
   - Creating new interaction terms
   - Standardization/Normalization

3. **Modeling**
   - Ridge, Lasso, ElasticNet
   - Gradient Boosting, XGBoost, LightGBM
   - Ensembling with Stacked Regressor

4. **Evaluation**
   - RMSE as primary metric
   - Cross-validation for model robustness

---

## 🧰 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost, LightGBM
- Kaggle Notebook

---

## 📈 Results

| Model | Cross-Validated RMSE |
|-------|----------------------|
| Ridge Regression | 0.1154 |
| Lasso | 0.1123 |
| XGBoost | 0.1132 |
| Stacked Model | **0.1097** ✅ |

---

## 🛠️ How to Run

1. Clone this repo:
```bash
git clone https://github.com/your-username/house-prices-advanced-regression.git
cd house-prices-advanced-regression
