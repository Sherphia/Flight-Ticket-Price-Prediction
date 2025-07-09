# Flight-Ticket-Price-Prediction

This repository contains a comprehensive machine learning solution for predicting airline ticket prices based on flight attributes such as airline, route, travel class, trip duration, and advance booking period.

---

## 📖 Project Summary

- **Problem Statement**: Predict flight prices using structured features from airline booking data.
- **Objective**: Train and compare regression models to identify the most accurate and generalizable predictor.
- **Dataset**: Provided via Kaggle (train/test split with missing values and categorical features).
- **Target Variable**: `price` (flight fare in INR)
- **Task Type**: Supervised Regression


## 🔍 Exploratory Data Analysis (EDA)

- Reviewed data types, shape, and missing values.
- Visualized distributions and outliers in key variables (`price`, `duration`, `days_left`).
- Extracted insights such as:
  - Business class fares are significantly higher and more variable.
  - Prices increase with flight duration and decrease with earlier bookings.
  - Airline and departure time influence price trends.

---

## 🧹 Data Cleaning and Preprocessing

- **Missing Values**:
  - Imputed categorical features using mode.
  - Imputed numeric features (`duration`, `days_left`) using median.
- **Outlier Handling**:
  - Removed extreme outliers (`price > ₹100,000`).
  - Capped flight durations at the 95th percentile.
- **Encoding**:
  - Label encoded travel `class`.
  - One-hot encoded categorical features (e.g., airline, stops, departure time).
- **Feature Scaling**:
  - Standardized `duration` and `days_left` using `StandardScaler`.

---

## ⚙️ Models and Techniques

The following models were implemented and evaluated:

| Model                  | Tuned | Library     |
|------------------------|-------|-------------|
| Linear Regression       | ❌    | scikit-learn |
| Ridge Regression        | ❌    | scikit-learn |
| Lasso Regression        | ❌    | scikit-learn |
| Decision Tree Regressor| ✅    | scikit-learn |
| Random Forest Regressor| ✅    | scikit-learn |
| Gradient Boosting      | ✅    | scikit-learn |
| AdaBoost Regressor     | ❌    | scikit-learn |
| XGBoost Regressor      | ✅    | XGBoost      |
| LightGBM Regressor     | ✅    | LightGBM     |

### Hyperparameter Tuning
GridSearchCV was used for:
- Random Forest
- Gradient Boosting
- Decision Tree
- XGBoost
- LightGBM

Evaluation Metrics:
- **R² Score**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

---

### 📊 Model Performance Comparison

The following table summarizes the performance of all 14 models evaluated on the validation set, using **R² Score**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**:

| Model                     | R² Score | RMSE | MAE  | Tuned? |
|---------------------------|----------|------|------|--------|
| Random Forest (Tuned)     | 0.9699   | 3920 | 2154 | ✅ Yes |
| Random Forest             | 0.9692   | 3963 | 2151 | ❌ No  |
| XGBoost (Tuned)           | 0.9692   | 3965 | 2293 | ✅ Yes |
| XGBoost                   | 0.9670   | 4105 | 2417 | ❌ No  |
| LightGBM (Tuned)          | 0.9667   | 4125 | 2442 | ✅ Yes |
| LightGBM                  | 0.9648   | 4236 | 2507 | ❌ No  |
| Gradient Boosting (Tuned) | 0.9634   | 4325 | 2569 | ✅ Yes |
| Decision Tree (Tuned)     | 0.9606   | 4485 | 2504 | ✅ Yes |
| Gradient Boosting         | 0.9483   | 5135 | 3090 | ❌ No  |
| Decision Tree             | 0.9460   | 5251 | 2482 | ❌ No  |
| AdaBoost                  | 0.9283   | 6051 | 3757 | ❌ No  |
| Linear Regression         | 0.9110   | 6740 | 4566 | ❌ No  |
| Ridge Regression          | 0.9110   | 6740 | 4566 | ❌ No  |
| Lasso Regression          | 0.9110   | 6741 | 4565 | ❌ No  |

📌 **Observation**:  
Tree-based ensemble models, especially after hyperparameter tuning (e.g., Random Forest, XGBoost, LightGBM), deliver superior performance compared to linear models. The tuned Random Forest model consistently outperforms others across all evaluation metrics and was selected for final predictions.

## 🏆 Best Model

| Metric        | Tuned Random Forest |
|---------------|---------------------|
| **R² Score**  | 0.9699              |
| **RMSE**      | ~3920               |
| **MAE**       | ~2154               |

The **Tuned Random Forest Regressor** outperformed all other models and was selected as the final model for generating predictions on the test dataset.

---

## 📈 Visualizations

- **Bar plots**: Model comparison by R², RMSE, MAE
- **Scatter plots**: Actual vs. predicted prices
- **EDA insights**:
  - Airline vs. Average Price
  - Travel Class vs. Price Distribution
  - Duration vs. Price by Class
  - Departure Time vs. Fare Trends

---

## 📤 Final Submission

- Applied identical preprocessing to test dataset.
- Predictions generated using the tuned Random Forest model.
- Saved as `submission.csv` for Kaggle evaluation.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn
  - XGBoost, LightGBM
- **Tools**: Jupyter Notebook, GridSearchCV

---
