from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "mara",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="topic_training_pipeline",
    description="Train unsupervised NMF topic model, log to W&B, and upload artifacts to GCS",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["training", "topic-modeling", "wandb", "gcs"],
) as dag:

    train_topic_model = BashOperator(
        task_id="train_topic_model",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.training.training_topic_model "
            "--run-date {{ ds }}"
        ),
    )

    upload_nmf_model = BashOperator(
        task_id="upload_nmf_model",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path models/topic_model/{{ ds }}/nmf_topic_model.joblib "
            "--gcs-prefix models/topic_model/{{ ds }}"
        ),
    )

    upload_vectorizer = BashOperator(
        task_id="upload_vectorizer",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path models/topic_model/{{ ds }}/tfidf_vectorizer.joblib "
            "--gcs-prefix models/topic_model/{{ ds }}"
        ),
    )

    upload_metadata = BashOperator(
        task_id="upload_metadata",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path models/topic_model/{{ ds }}/topic_metadata.json "
            "--gcs-prefix models/topic_model/{{ ds }}"
        ),
    )

    upload_training_assignments = BashOperator(
        task_id="upload_training_assignments",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path data/predictions/topic_assignments_training_{{ ds }}.csv "
            "--gcs-prefix predictions/training"
        ),
    )

    (
        train_topic_model
        >> [
            upload_nmf_model,
            upload_vectorizer,
            upload_metadata,
            upload_training_assignments,
        ]
    )