import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load dataset
data = pd.read_csv("data/student_data.csv")


# Features
features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks"
]

X = data[features]
y = data["result"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load trained model
model = joblib.load("models/student_model.pkl")


# Prediction
predictions = model.predict(X_test)


# Metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    pos_label="Pass",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="Pass",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="Pass",
    zero_division=0
)


# Display results
print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))