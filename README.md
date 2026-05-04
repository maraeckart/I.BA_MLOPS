# I.BA_MLOPS

# Live European News Monitoring System

This project builds a live machine learning pipeline for European news monitoring.
It collects RSS news headlines, preprocesses them, creates basic text and time-based features,
and later classifies each headline into a topic category.

http://localhost:8080
guardian-news-airflow  | Simple auth manager | Password for user 'admin': 7EvFCZhCUKKbET8P


headline_clean → TF-IDF → Logistic Regression → topic/category prediction

needs to be updaated!!!!!!

Docker:
docker compose run --rm ingest-news
docker compose run --rm preprocess-news
docker compose up airflow