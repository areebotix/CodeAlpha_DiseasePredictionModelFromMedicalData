import joblib
import numpy as np

model = joblib.load("../models/heart_model.pkl")

def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    return "Disease Detected" if prediction == 1 else "No Disease"