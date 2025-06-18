# Telecom Customer Churn Prediction

## 🚀 Overview  
An end-to-end machine-learning pipeline to predict customer churn for a telecom provider.  
Key steps: data loading & validation, exploratory analysis (including skew inspection), PCA for dimensionality reduction, and comparison of three classifiers—ultimately selecting an SVM with RBF kernel for the best cross-validated accuracy. Also, saving the models using joblib to use them again.

## 📦 Dataset  
- **Source**: [Kaggle Telecom Churn Dataset by barun2104](https://www.kaggle.com/datasets/barun2104/telecom-churn/data)  
- **File**: `telecom_churn.csv`  
- **Shape**: 3 333 rows × 11 columns  
- **Features**:  
  - **Continuous**: `AccountWeeks`, `DataUsage`, `DayMins`, `DayCalls`, `MonthlyCharge`, `OverageFee`, `RoamMins`  
  - **Binary**: `DataPlan`, `ContractRenewal`  
  - **Count**: `CustServCalls`  
- **Target**: `Churn` (0 = stay, 1 = leave)  
- **Original class distribution**:  
  - 0 (stay): 2 850  
  - 1 (churn):   483  

## 🔍 Exploratory Data Analysis  
1. **Integrity checks**  
   - No missing values  
   - No duplicate rows  
2. **Feature distributions**  
   - Plotted histograms & boxplots for all numeric fields  
   - Discovered `DataUsage` is heavily right-skewed  
3. **Skew correction experiments**  
   - Temporarily created `DataUsage_log`, `DataUsage_boxcox`, and `DataUsage_yj` to compare log1p/Box-Cox/Yeo–Johnson transforms  
   - Visualized each transform but ultimately dropped these columns and retained the original `DataUsage` for modeling  

## 🛠 Preprocessing  
- **Features used**: all original columns except `Churn`  
- **Pipeline steps**:  
  1. `StandardScaler()` – zero-mean, unit-variance scaling  
  2. `PCA(n_components=0.95)` – retain 95% of total variance  

## ⚙️ Modeling & Cross-Validation  
Wrapped each classifier in the same preprocessing pipeline and ran **5-fold stratified CV** (shuffle, `random_state=42`):

| Model                     | CV Accuracy (mean ± std) |
|---------------------------|--------------------------|
| Logistic Regression       | 0.860 ± 0.005            |
| Random Forest (100 trees) | **0.915 ± 0.004**            |
| **SVM (RBF kernel)**      | **0.915 ± 0.006**        |

> **Winner:** The SVM & Random Forest achieved the highest average accuracy with almost similar variance.
