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
    dag_id="guardian_news_pipeline",
    description="Daily Guardian RSS ingestion, preprocessing, and GCS upload pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 4, 29, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["news", "guardian", "mlops", "gcs"],
) as dag:

    collect_news = BashOperator(
        task_id="collect_news",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.data_ingestion.collect_news "
            "--run-date {{ ds }}"
        ),
    )

    upload_raw_to_gcs = BashOperator(
        task_id="upload_raw_to_gcs",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path data/raw/rss/live/raw_guardian_news_{{ ds }}.csv "
            "--gcs-prefix raw/rss/live"
        ),
    )

    preprocess_news = BashOperator(
        task_id="preprocess_news",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.preprocessing.preprocess_news "
            "--run-date {{ ds }}"
        ),
    )

    upload_processed_to_gcs = BashOperator(
        task_id="upload_processed_to_gcs",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path data/processed/live/processed_guardian_news_{{ ds }}.csv "
            "--gcs-prefix processed/live"
        ),
    )

    collect_news >> upload_raw_to_gcs >> preprocess_news >> upload_processed_to_gcs