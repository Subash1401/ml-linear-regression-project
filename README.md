# Salary Prediction Using Machine Learning

## 📌 Project Overview

This project predicts employee salaries using Machine Learning and Linear Regression. It uses employee details such as Age, Experience, City, and Education to estimate salary.

The project covers the complete Machine Learning workflow, from data preprocessing to model evaluation.

## 🎯 Problem Statement

To build a Machine Learning model that predicts employee salaries based on relevant employee information and identifies patterns between employee attributes and salary.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Data Analysis:** Pandas
* **Machine Learning:** Scikit-learn
* **Data Preprocessing:** StandardScaler, One-Hot Encoding
* **Model:** Linear Regression
* **Evaluation Metrics:** Mean Absolute Error (MAE), R² Score

## 📊 Dataset Description

The dataset contains the following columns:

| Feature    | Description                          |
| ---------- | ------------------------------------ |
| Age        | Employee's age                       |
| Experience | Years of work experience             |
| City       | Employee's city                      |
| Education  | Employee's educational qualification |
| Salary     | Target variable to predict           |

The dataset is created manually using Python and contains sample employee records for learning and experimentation.

## ⚙️ Project Workflow

1. **Data Creation:** Create a sample employee dataset using Pandas.
2. **Exploratory Data Analysis (EDA):** Inspect the dataset using `head()`, `shape`, `info()`, and `describe()`.
3. **Missing Value Handling:** Fill missing Age values with the mean and missing City values with the mode.
4. **Duplicate Removal:** Identify and remove duplicate records.
5. **Outlier Detection:** Use the Interquartile Range (IQR) method to identify and remove salary outliers.
6. **Categorical Encoding:** Convert City and Education columns into numerical features using One-Hot Encoding with `pd.get_dummies()`.
7. **Feature and Target Separation:** Separate input features (X) and target salary (y).
8. **Train-Test Split:** Split the dataset into training and testing sets.
9. **Feature Scaling:** Apply StandardScaler to scale input features.
10. **Model Training:** Train a Linear Regression model using the training data.
11. **Prediction:** Predict salaries for the test data.
12. **Model Evaluation:** Evaluate predictions using MAE and R² Score.

## 🤖 Machine Learning Algorithm

**Linear Regression**

Linear Regression learns the relationship between input features and a numerical target variable to predict salary values.

## 📈 Model Evaluation

The model is evaluated using:

* **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted salaries. Lower values are generally better.
* **R² Score:** Measures how well the model explains variations in the target values. A higher score generally indicates a better fit.

Actual evaluation results depend on the train-test split and the sample dataset.

## 💻 Installation and Execution

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd salary-prediction-ml
```

### 3. Install Required Libraries

```bash
pip install pandas scikit-learn
```

### 4. Run the Python Program

```bash
python salary_prediction.py
```

## 📚 Key Learning Outcomes

* Data cleaning and preprocessing using Pandas
* Handling missing values and duplicate records
* Detecting outliers using the IQR method
* Encoding categorical variables
* Splitting data into training and testing sets
* Feature scaling using StandardScaler
* Training a supervised Machine Learning model
* Evaluating regression models using MAE and R² Score

## 🌍 Real-World Applications

* Employee salary estimation
* HR analytics and workforce planning
* Understanding salary patterns
* Exploring relationships between experience, education, and compensation

**Note:** This project uses a small, manually created sample dataset for educational purposes. Its predictions should not be used to determine real-world employee compensation without representative data, proper validation, and fairness checks.

## 👨‍💻 Author

**Subash S**

Aspiring AI/ML Engineer | Learning Machine Learning and Generative AI
