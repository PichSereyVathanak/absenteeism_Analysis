# Absenteeism Prediction (Streamlit + ML)

This project predicts whether an employee has excessive absenteeism using a trained logistic regression model and a Streamlit web interface. It includes data preprocessing, feature engineering, model training, model/scaler serialization, and a front-end form to make real-time predictions.

## 🚀 Project Overview

- `notebook/1_data_preprocessing.ipynb`: data cleanup, reason grouping (Group 1-4 by clinical codes), date features, and final prepared dataset (`clean_Absenteeism_data.csv`).
- `notebook/2_machine_learning.ipynb`: model training pipeline, custom scaler (`CustomScaler`), logistic regression, and model evaluation.
- `notebook/absenteeism_module.py`: reusable model wrapper class (`absenteeism_model`) that loads the saved model and scaler and performs prediction on new data.
- `streamlit_app.py`: user-facing UI via Streamlit, with input fields, reason-type explanation, date picker, and prediction output.
- `requirements.txt`: runtime dependencies for Python environment.

## 🧩 Requirement (dependencies)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 🛠️ Setup

1. Clone project
2. Create and activate venv (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure you have the files:
   - `notebook/model` (trained logistic regression pickle)
   - `notebook/scaler` (trained custom scaler pickle)

## ▶️ Running Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open shown URL (e.g., http://localhost:8501).

## 🧾 Usage

- Select absence reason type (Group 1-4; click “Read More” for details)
- Select a date (month extracted from date automatically)
- Provide employee details (age, children, pets, expenses, BMI, education)
- Click `Predict Absenteeism Risk`
- View probability and classification

## ♻️ File structure

```
requirements.txt
streamlit_app.py
README.md
.gitignore
notebook/
  absenteism_module.py
  1_data_preprocessing.ipynb
  2_machine_learning.ipynb
  model
  scaler
  __init__.py
data/
  raw/
  preprocessed/
```