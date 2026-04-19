import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/processed/cleaned_heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# Evaluation
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Save model
joblib.dump(model, "../models/heart_model.pkl")

print("Model saved ✔️")