from flask import Flask, render_template, request
import joblib


app = Flask(__name__)


# Load trained model
model = joblib.load("models/student_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_score = float(request.form["assignment_score"])
    internal_marks = float(request.form["internal_marks"])


    input_data = [[
        study_hours,
        attendance,
        previous_marks,
        assignment_score,
        internal_marks
    ]]


    prediction = model.predict(input_data)[0]


    return render_template(
        "result.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )