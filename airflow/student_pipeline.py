from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def preprocess_data():
    print("Step 1: Preprocessing student data")


def train_model():
    print("Step 2: Training ML model")


def evaluate_model():
    print("Step 3: Evaluating ML model")


def log_to_mlflow():
    print("Step 4: Logging results to MLflow")


with DAG(
    dag_id="student_performance_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    mlflow = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow
    )


    preprocess >> train >> evaluate >> mlflow