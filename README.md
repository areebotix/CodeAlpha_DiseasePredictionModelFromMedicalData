# ❤️ Heart Disease Prediction System (Machine Learning + Explainable AI)

## 📌 Overview
This project builds a machine learning system to predict the risk of heart disease using clinical parameters. It includes model training, evaluation, and an explainable AI component to interpret predictions.

The goal is to demonstrate an end-to-end ML pipeline with real-world healthcare data.

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP (Explainable AI)
- Streamlit (Web App)

---

## 📊 Dataset
- UCI Heart Disease Dataset  
- Features: age, cholesterol, blood pressure, chest pain type, etc.  
- Target: presence/absence of heart disease  

---

## 🔁 Workflow
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training:
  - Logistic Regression  
  - Support Vector Machine  
  - Random Forest  
  - XGBoost (Best Model)  
- Hyperparameter Tuning  
- Model Evaluation  
- Explainability using SHAP  

---

## 🏆 Best Model Performance
- XGBoost achieved the highest accuracy (~83% test accuracy)
- Reduced overfitting using tuned parameters

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision / Recall / F1-score  
- Confusion Matrix  
- ROC Curve  
- Feature Importance Analysis  

---

## 🧠 Explainable AI (SHAP)
SHAP was used to interpret model predictions and identify which medical features contribute most to heart disease risk.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
Run training:

python src/train_model.py

Run prediction:

python src/predict.py

Run Streamlit app:

streamlit run src/app.py
📁 Project Structure
data/
models/
notebooks/
src/
outputs/
README.md
requirements.txt
⚠️ Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool.

👨‍💻 Author
Areeba Khan
---
