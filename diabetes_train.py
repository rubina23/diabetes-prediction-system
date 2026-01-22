
# **Diabetes Prediction System**

# Steps:

##**1. Data Loading**

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score



# Dataset load
df = pd.read_csv("diabetes.csv")

# Display first few rows and shape
print(df.head())
print("Dataset shape:", df.shape)


## **2. Data Preprocessing**
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



## **3. Pipeline Creation**

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])



## **4. Model Training**

# Train the pipeline on the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

## **5. Cross-Validation**
# Apply 5-fold cross-validation on the training set

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Cross-Validation Mean:", scores.mean())
print("Cross-Validation Std:", scores.std())


## **6. Hyperparameter Tuning**

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

## **7. Best Model Selection**
# Select the best model from GridSearchCV
best_model = grid.best_estimator_
print("Final Best Model:", best_model)

## **9. Model Performance Evaluation**
# Predict on the test set
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


"""## **Save Model**"""
# Save the pipeline instead of only model

import pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

"""**See rest of task on app.py file**"""
