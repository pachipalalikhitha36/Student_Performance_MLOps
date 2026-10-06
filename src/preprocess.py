import pandas as pd


FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks"
]


def load_data():
    data = pd.read_csv("data/student_data.csv")
    return data


def prepare_data():
    data = load_data()

    X = data[FEATURES]
    y = data["result"]

    return X, y


if __name__ == "__main__":
    X, y = prepare_data()

    print("Data loaded successfully!")
    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nDataset shape:", X.shape)