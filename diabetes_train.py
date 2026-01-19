
# **Diabetes Prediction System**

# Steps:

##**1. Data Loading (5 Marks)**
#Load the chosen dataset into your environment and display the first few rows along with the shape to verify correctness.


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score



# Dataset load
df = pd.read_csv("diabetes.csv")

# Display first few rows and shape
print(df.head())
print("Dataset shape:", df.shape)

"""## **2. Data Preprocessing (10 Marks)**
Perform and document at least 5 distinct preprocessing steps (e.g., handling missing values, encoding, scaling, outlier detection, feature engineering).
"""

# Check missing values
print(df.isnull().sum())

# Outlier handling (example: remove extreme BMI values)
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['BMI'] >= Q1 - 1.5*IQR) & (df['BMI'] <= Q3 + 1.5*IQR)]

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Outcome', axis=1))

X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['Outcome']

"""## **3. Pipeline Creation (10 Marks)**
Construct a standard Machine Learning pipeline that integrates preprocessing and the model
"""

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

"""## **4. Primary Model Selection (5 Marks)**
Choose a suitable algorithm and justify why this specific model was selected for the dataset.    

**Answer:**  Logistic Regression is a good first choice because:
*   Binary classification problem (Outcome: 0/1)
*   Easy to interpret coefficients
*   Baseline model for healthcare datasets

## **5. Model Training (10 Marks)**
Train your selected model using the training portion of your dataset.
"""

# Train the pipeline on the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

"""## **6. Cross-Validation (10 Marks)**
Apply Cross-Validation  to assess robustness and report the average score with standard deviation.
"""

# Apply 5-fold cross-validation on the training set

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Cross-Validation Mean:", scores.mean())
print("Cross-Validation Std:", scores.std())

"""## **7. Hyperparameter Tuning (10 Marks)**
Optimize your model using search methods displaying both the parameters tested and the best results found.
"""

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

"""## **8. Best Model Selection (10 Marks)**
Select  the final best-performing model based on the hyperparameter tuning results.
"""

# Select the best model from GridSearchCV
best_model = grid.best_estimator_

print("Final Best Model:", best_model)

"""## **9. Model Performance Evaluation (10 Marks)**
Evaluate the model on the test set and print comprehensive metrics suitable for the problem type.
"""

# Predict on the test set
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""## **Save & Load Model**"""

# Save the pipeline instead of only model
import pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

"""**See rest of task on app.py file**"""