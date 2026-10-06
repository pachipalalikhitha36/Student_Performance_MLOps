import joblib
import pandas as pd


def test_model_exists():

    model = joblib.load("models/student_model.pkl")

    assert model is not None


def test_model_prediction():

    model = joblib.load("models/student_model.pkl")

    data = pd.DataFrame([
        {
            "study_hours": 6,
            "attendance": 85,
            "previous_marks": 70,
            "assignment_score": 75,
            "internal_marks": 72
        }
    ])

    prediction = model.predict(data)

    assert prediction[0] in ["Pass", "Fail"]