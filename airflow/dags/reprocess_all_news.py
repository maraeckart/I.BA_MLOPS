from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "mara",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def reprocess_all_news_files() -> None:
    from src.preprocessing.preprocess_news import preprocess_news, save_processed_news
    from src.storage.upload_to_gcs import upload_file_to_gcs

    jobs = []

    # Live RSS files
    raw_rss_dir = Path("/app/data/raw/rss/live")
    processed_live_dir = Path("/app/data/processed/live")
    processed_live_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in sorted(raw_rss_dir.glob("raw_news_*.csv")):
        run_date = raw_file.stem.replace("raw_news_", "")
        output_path = processed_live_dir / f"processed_news_{run_date}.csv"

        jobs.append(
            {
                "input_path": raw_file,
                "output_path": output_path,
                "gcs_prefix": "processed/live",
            }
        )

    # API backfill files
    raw_api_dir = Path("/app/data/raw/api")
    processed_api_dir = Path("/app/data/processed/api")
    processed_api_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in sorted(raw_api_dir.glob("raw_guardian_api_*.csv")):
        output_name = raw_file.name.replace("raw_", "processed_")
        output_path = processed_api_dir / output_name

        jobs.append(
            {
                "input_path": raw_file,
                "output_path": output_path,
                "gcs_prefix": "processed/backfill",
            }
        )

    if not jobs:
        raise FileNotFoundError("No raw RSS or API files found for reprocessing.")

    print(f"Found {len(jobs)} files to reprocess.")

    for job in jobs:
        print(f"Preprocessing: {job['input_path']}")

        processed_df = preprocess_news(job["input_path"])

        saved_path = save_processed_news(
            news_df=processed_df,
            output_path=job["output_path"],
        )

        print(f"Saved processed file: {saved_path}")

        upload_file_to_gcs(
            local_path=saved_path,
            gcs_prefix=job["gcs_prefix"],
        )


with DAG(
    dag_id="reprocess_all_news",
    description="One-time reprocessing of all raw Guardian RSS/API files with topic_text column",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["news", "preprocessing", "backfill", "gcs"],
) as dag:

    reprocess_all = PythonOperator(
        task_id="reprocess_all_news_files",
        python_callable=reprocess_all_news_files,
    )