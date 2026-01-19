# 🩺 Diabetes Prediction System

A Machine Learning project to predict whether a patient has diabetes based on medical attributes.  
This project uses the **Pima Indians Diabetes Dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## 📂 Project Structure

```
├── app.py                  # Gradio web app
├── diabetes.csv            # Dataset
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── diabetes_model.pkl      # Saved model
└── diabetes_train.ipynb    # model trained

```

---

## 🎯 Tasks Completed

1. **Data Loading** – Loaded dataset, displayed shape & first few rows.
2. **Data Preprocessing** – Handled missing values, outliers, scaling, encoding, feature engineering.
3. **Pipeline Creation** – Built ML pipeline combining preprocessing + model.
4. **Primary Model Selection** – Logistic Regression chosen for binary classification.
5. **Model Training** – Trained on training set.
6. **Cross-Validation** – 5-fold CV applied, reported mean ± std.
7. **Hyperparameter Tuning** – GridSearchCV used to optimize parameters.
8. **Best Model Selection** – Selected final tuned model.
9. **Model Performance Evaluation** – Accuracy, confusion matrix, classification report.
10. **Web Interface** – Gradio app built for user-friendly predictions.
11. **Deployment** – Deployed to Hugging Face Spaces.

---

## ⚙️ Tech Stack

- **Python 3.9+**
- **Pandas, NumPy**
- **Scikit-learn**
- **Gradio**
- **Hugging Face Spaces**

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system

```

**Install dependencies**

```

pip install -r requirements.txt

```

**Run Gradio app**

```

python app.py

```

---

# 🌐 Live Demo

👉 Try the App on Hugging Face Spaces: https://huggingface.co/spaces/rubina25/Diabetes-Prediction-System

---

# 📊 Sample Input/Output

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Prediction   |
| ----------- | ------- | ------------- | ------------- | ------- | ---- | ------------------------ | --- | ------------ |
| 2           | 120     | 70            | 25            | 80      | 28.5 | 0.45                     | 35  | Not Diabetic |
| 6           | 165     | 90            | 35            | 200     | 33.2 | 0.75                     | 50  | Diabetic     |

---

# 📈 Results

- Cross-Validation Accuracy: ~0.77 ± 0.04

- Test Accuracy: ~0.78

- Metrics: Precision, Recall, F1-score reported in classification report.

---

# 👨‍💻 Author

Name: **Rubina Begum**

Email: your.email@example.com

GitHub: your-username (github.com in Bing)

---

# 📌 Notes

This project is for educational purposes and demonstrates end-to-end ML workflow.

Not intended for real medical diagnosis.
