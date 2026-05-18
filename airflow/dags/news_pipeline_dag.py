from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator


MODEL_DATE = "2026-05-12"
ROLLING_WINDOW_DAYS = 3

default_args = {
    "owner": "mara",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="news_live_pipeline",
    description="Daily news ingestion, preprocessing, topic prediction, and GCS upload pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["news", "mlops", "gcs", "topic-modeling"],
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
            "--local-path data/raw/rss/live/raw_news_{{ ds }}.csv "
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
            "--local-path data/processed/live/processed_news_{{ ds }}.csv "
            "--gcs-prefix processed/live"
        ),
    )

    train_topic_model = BashOperator(
    task_id="train_topic_model",
    bash_command=(
        "cd /app && "
        "mkdir -p models/topic_model/{{ ds }} && "
        "PYTHONPATH=/app "
        "python -m src.training.training_topic_model "
        "--processed-dir data/processed/live "
        "--start-date {{ macros.ds_add(ds, -" + str(ROLLING_WINDOW_DAYS - 1) + ") }} "
        "--end-date {{ ds }} "
        "--model-dir models/topic_model/{{ ds }}"
    ),
)

    predict_topics = BashOperator(
    task_id="predict_topics",
    bash_command=(
        "cd /app && "
        "mkdir -p data/predictions/live && "
        "PYTHONPATH=/app "
        "python -m src.inference.predict_topics "
        "--input-path data/processed/live/processed_news_{{ ds }}.csv "
        "--model-dir models/topic_model/{{ ds }} "
        "--output-path data/predictions/live/topic_predictions_{{ ds }}.csv"
    ),
)

    upload_predictions_to_gcs = BashOperator(
        task_id="upload_predictions_to_gcs",
        bash_command=(
            "cd /app && "
            "PYTHONPATH=/app "
            "python -m src.storage.upload_to_gcs "
            "--local-path data/predictions/live/topic_predictions_{{ ds }}.csv "
            "--gcs-prefix predictions/live"
        ),
    )

    (
        collect_news
        >> upload_raw_to_gcs
        >> preprocess_news
        >> upload_processed_to_gcs
        >> train_topic_model
        >> predict_topics
        >> upload_predictions_to_gcs
    )