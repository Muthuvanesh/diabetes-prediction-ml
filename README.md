# 🩺 Diabetes Prediction Using Machine Learning

<div align="center">
  <img src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1400&q=80" alt="Healthcare and medical analytics illustration" width="100%" />
</div>

## 📌 Project Overview

This project builds a machine learning pipeline to predict whether a patient is likely to have diabetes based on medical health indicators such as glucose, BMI, blood pressure, insulin, age, and more.

The notebook performs end-to-end data analysis, preprocessing, feature engineering, model tuning, evaluation, and prediction using a real-world diabetes dataset.

> ⚠️ This project is intended for educational and research purposes only and is not a substitute for professional medical advice or diagnosis.

---

## 🎯 Objectives

- Perform exploratory data analysis (EDA)
- Detect missing values and invalid zero entries
- Analyze class distribution and patient risk patterns
- Engineer new predictive features
- Train and optimize a strong classification model
- Evaluate model performance using real metrics
- Save the final model and make new predictions

---

## 🧬 Dataset Information

The project uses the `diabetes.csv` dataset containing 768 patient records with medical attributes.

### Features

| Feature | Description |
| --- | --- |
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree score |
| Age | Patient age |
| Outcome | Target label: 0 = No Diabetes, 1 = Diabetes |

### Target Variable

- `0` → No Diabetes
- `1` → Diabetes

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Pickle

---

## 🧪 Project Workflow

### 1. Data Loading and Inspection

The notebook loads the dataset and checks:

- `df.head()`
- `df.shape`
- `df.info()`
- `df.describe()`
- missing values
- zero values across columns

### 2. Data Cleaning

The dataset contains invalid zero values in fields such as:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

These are converted to `NaN` before model training so the imputer can handle them correctly.

### 3. Exploratory Data Analysis

The notebook explores:

- target class balance
- diabetes outcome distribution
- patient risk-related patterns
- relationships between features and the outcome

### 4. Feature Engineering

The project creates additional predictive features such as:

- `Glucose_BMI`
- `Glucose_Age`
- `BMI_Age`
- `RiskScore`
- `High_Glucose`
- `Obese`
- `Older`

This increases the model’s ability to learn meaningful patterns from the medical data.

### 5. Train-Test Split

The dataset is split using:

- `train_test_split`
- `test_size=0.2`
- `random_state=42`
- `stratify=y`

This ensures the training and test sets maintain a balanced class distribution.

### 6. Model Training

The notebook uses a machine learning pipeline with:

- `KNNImputer` for missing value handling
- `XGBClassifier` for classification
- `GridSearchCV` for hyperparameter tuning
- `StratifiedKFold` cross-validation

The model is optimized to improve predictive performance on the diabetes classification task.

### 7. Model Evaluation

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### 8. Prediction for a New Patient

After training, the model is saved and loaded again to predict outcomes for a new patient using medical values.

---

## 📊 Model Performance

The notebook reports the model’s performance on the held-out test set using real evaluation metrics.

The final setup demonstrates a pipeline-based approach with:

- feature engineering
- KNN imputation
- XGBoost classification
- cross-validated tuning
- confusion matrix analysis

Typical results from the notebook show strong predictive performance and around 74% test accuracy, with a confidence score for patient classification.

---

## 🧠 Example Prediction

```python
import pickle
import pandas as pd

with open('diabetes_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

new_patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 120,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50,
    'Glucose_BMI': 148 * 33.6,
    'Glucose_Age': 148 * 50,
    'BMI_Age': 33.6 * 50,
    'RiskScore': (148/100) + (33.6/30) + (50/40),
    'High_Glucose': 1,
    'Obese': 1,
    'Older': 1
}])

prediction = model.predict(new_patient)[0]
print('Diabetes' if prediction == 1 else 'No Diabetes')
```

---

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetes-prediction-ml.git
```

### 2. Open the project folder

```bash
cd diabetes-prediction-ml
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `diabetes_classification_ml.ipynb` and run the cells in order.

---

## 📁 Project Structure

```text
diabetes-prediction-ml/
├── diabetes_classification_ml.ipynb
├── diabetes.csv
├── diabetes_xgboost.pkl
├── README.md
└── .gitignore
```

---

## ✅ Key Learnings

This project demonstrates practical knowledge in:

- data cleaning and validation
- exploratory data analysis
- feature engineering
- model tuning with cross-validation
- evaluating classification metrics
- understanding confusion matrices
- saving and reusing machine learning models
- making real predictions from patient data

---

## 🔮 Future Improvements

- tune hyperparameters more deeply for better accuracy
- compare XGBoost with other classifiers
- create a web-based prediction app
- add model explainability using feature importance plots
- deploy the model for live inference

---

## 📘 Disclaimer

This project is developed for learning and demonstration purposes only. Prediction outcomes are based on historical data and are not a clinical diagnosis or medical recommendation.
