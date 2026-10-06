import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
DATA_PATH = "data/student_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# Features and target
features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks"
]

X = data[features]
y = data["result"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


# Train model
print("Training model...")

model.fit(X_train, y_train)

print("Model training completed!")


# Prediction
predictions = model.predict(X_test)


# Evaluation
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
print("\nModel Performance")
print("-------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# Create models folder
os.makedirs("models", exist_ok=True)


# Save model
MODEL_PATH = "models/student_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)