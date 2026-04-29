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
    description="Daily Guardian RSS ingestion and preprocessing pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 4, 29, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["news", "guardian", "mlops"],
) as dag:

    collect_news = BashOperator(
        task_id="collect_news",
        bash_command=(
            "cd /app && "
            "python -m src.data_ingestion.collect_news "
            "--run-date {{ ds }}"
        ),
    )
    preprocess_news = BashOperator(
        task_id="preprocess_news",
        bash_command=(
            "cd /app && "
            "python -m src.preprocessing.preprocess_news "
            "--run-date {{ ds }}"
        ),
    )
    collect_news >> preprocess_news