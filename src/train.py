import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# -----------------------------
# 1. Load dataset
# -----------------------------

data = pd.read_csv("data/student_data.csv")


# -----------------------------
# 2. Select features
# -----------------------------

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks"
]

X = data[features]
y = data["result"]


# -----------------------------
# 3. Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 4. Start MLflow
# -----------------------------

mlflow.set_experiment("Student_Performance_Prediction")


with mlflow.start_run():

    # -----------------------------
    # 5. Create model
    # -----------------------------

    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )


    # -----------------------------
    # 6. Train model
    # -----------------------------

    model.fit(X_train, y_train)


    # -----------------------------
    # 7. Prediction
    # -----------------------------

    predictions = model.predict(X_test)


    # -----------------------------
    # 8. Evaluation
    # -----------------------------

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


    # -----------------------------
    # 9. Print results
    # -----------------------------

    print("\nModel Training Completed!")
    print("-----------------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


    # -----------------------------
    # 10. Log parameters
    # -----------------------------

    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)


    # -----------------------------
    # 11. Log metrics
    # -----------------------------

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


    # -----------------------------
    # 12. Save model
    # -----------------------------
# -----------------------------
# 12. Save model
# -----------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/student_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:", model_path)


# -----------------------------
# 13. Log model to MLflow
# -----------------------------

mlflow.sklearn.log_model(
    model,
    "student_performance_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

print("Model logged to MLflow successfully!")