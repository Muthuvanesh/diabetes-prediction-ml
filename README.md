# Diabetes Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning classification algorithms to predict whether a person is likely to have diabetes based on medical health parameters.

The project includes data exploration, data preprocessing, visualization, model training, performance evaluation, and prediction for new patient data.

The objective is to understand how different classification algorithms perform on a diabetes dataset and use the trained models to make predictions.

**Note:** This project is intended for educational purposes and is not a substitute for professional medical diagnosis.

---

## Project Objectives

* Perform Exploratory Data Analysis (EDA).
* Identify missing values, duplicate records, and zero values.
* Analyze diabetes outcome distribution.
* Explore diabetes outcomes across different age groups.
* Preprocess numerical and categorical features.
* Train multiple Machine Learning classification models.
* Compare model performance using evaluation metrics.
* Visualize model performance using a confusion matrix.
* Predict diabetes outcomes for new patient inputs.

---

## Dataset Information

The project uses a diabetes dataset containing medical information about patients.

### Features

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| Pregnancies              | Number of pregnancies                      |
| Glucose                  | Plasma glucose concentration               |
| BloodPressure            | Diastolic blood pressure                   |
| SkinThickness            | Triceps skin fold thickness                |
| Insulin                  | 2-hour serum insulin                       |
| BMI                      | Body Mass Index                            |
| DiabetesPedigreeFunction | Diabetes pedigree function                 |
| Age                      | Patient's age                              |
| Outcome                  | Target variable indicating diabetes status |

### Target Variable

* `0` – No Diabetes
* `1` – Diabetes

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Loading

Loaded the diabetes dataset using Pandas.

```python
df = pd.read_csv("diabetes.csv")
```

### 2. Data Exploration

Performed initial dataset inspection using:

* `df.head()`
* `df.tail()`
* `df.shape`
* `df.columns`
* `df.info()`
* `df.duplicated()`
* `df.isnull().sum()`

Also examined zero values across the dataset.

### 3. Exploratory Data Analysis (EDA)

Performed visual analysis to understand the distribution of diabetes outcomes and patient age groups.

Visualizations include:

* Diabetes outcome distribution using a count plot.
* Diabetes outcome comparison across age groups.
* Exact value labels displayed on charts.

### 4. Data Preprocessing

* Separated input features (X) and target variable (y).
* Identified numerical and categorical features.
* Applied StandardScaler to numerical features.
* Applied OneHotEncoder to categorical features.
* Used ColumnTransformer and Pipeline to organize preprocessing and model training.

### 5. Train-Test Split

Split the dataset into training and testing sets.

* Training data: 80%
* Testing data: 20%
* Random state: 42
* Stratified splitting to preserve the target class distribution.

### 6. Machine Learning Models

Trained and evaluated the following classification algorithms:

**Logistic Regression**

A classification algorithm used to estimate the probability of a binary outcome.

**Decision Tree Classifier**

A tree-based algorithm that makes predictions using a sequence of decision rules.

**Random Forest Classifier**

An ensemble learning algorithm that combines multiple decision trees to make predictions.

---

## Model Evaluation

The models are evaluated using the following metrics:

| Metric    | Purpose                                                           |
| --------- | ----------------------------------------------------------------- |
| Accuracy  | Measures overall correct predictions                              |
| Precision | Measures how many predicted positive cases were actually positive |
| Recall    | Measures how many actual positive cases were correctly identified |
| F1-Score  | Harmonic mean of precision and recall                             |

The evaluation results are displayed in a comparison table to examine the performance of the three algorithms.

### Confusion Matrix

A confusion matrix is generated for the Random Forest model to visualize:

* True Positives
* True Negatives
* False Positives
* False Negatives

This helps examine the types of correct and incorrect predictions made by the model.

---

## Prediction for a New Person

The project includes an interactive prediction feature that accepts medical information from a new person.

The user enters:

* Pregnancies
* Glucose level
* Blood pressure
* Skin thickness
* Insulin level
* BMI
* Diabetes pedigree function
* Age

The input is converted into a Pandas DataFrame and passed to the trained model.

The model returns one of the following outputs:

* **Diabetes**
* **No Diabetes**

The project also visualizes estimated class probabilities using a bar chart.

---

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-prediction-ml.git
```

### Step 2: Navigate to the Project Folder

```bash
cd diabetes-prediction-ml
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Run the Notebook

```bash
jupyter notebook
```

Open `diabetes_classification_ml.ipynb` and execute the cells sequentially.

Ensure that `diabetes.csv` is available in the expected working directory.

---

## Project Structure

```text
diabetes-prediction-ml/
│
├── diabetes_classification_ml.ipynb
├── diabetes.csv
├── README.md
└── .gitignore
```

---

## Key Learnings

Through this project, I gained practical experience in:

* Data cleaning and validation.
* Exploratory Data Analysis.
* Data visualization using Matplotlib and Seaborn.
* Feature preprocessing using Scikit-learn.
* Training classification algorithms.
* Model evaluation and comparison.
* Confusion matrix interpretation.
* Making predictions using new input data.
* Building a complete Machine Learning workflow.

---

## Future Improvements

* Perform hyperparameter tuning to optimize model performance.
* Explore additional classification algorithms.
* Improve handling of zero values representing potentially missing medical measurements.
* Apply cross-validation for more robust model evaluation.
* Develop a web application using Django or Flask.
* Deploy the trained model for interactive predictions.

---

## Disclaimer

This project is developed for learning and demonstration purposes only. Predictions are based on the dataset and trained model and should not be interpreted as medical advice or a clinical diagnosis.
