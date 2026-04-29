import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜßéèàçÉÈÀÇ\s]", "", text)
    text = text.strip()
    return text


def clean_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df = news_df.dropna(subset=["headline"])
    news_df = news_df.drop_duplicates(subset=["headline", "url"])

    news_df["description"] = news_df["description"].fillna("")

    news_df["headline_clean"] = news_df["headline"].apply(clean_text)
    news_df["description_clean"] = news_df["description"].apply(clean_text)
    return news_df


def add_text_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["headline_length"] = news_df["headline_clean"].str.len()
    news_df["headline_word_count"] = news_df["headline_clean"].str.split().str.len()

    news_df["description_length"] = news_df["description_clean"].str.len()
    news_df["description_word_count"] = news_df["description_clean"].str.split().str.len()

    return news_df


def add_time_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["published_at_parsed"] = pd.to_datetime(
        news_df["published_at"],
        errors="coerce",
        utc=True,
    )

    news_df["publication_hour"] = news_df["published_at_parsed"].dt.hour
    news_df["publication_day_of_week"] = news_df["published_at_parsed"].dt.day_name()
    return news_df


def add_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = add_text_features(news_df)
    news_df = add_time_features(news_df)
    return news_df


def get_raw_file_for_date(run_date: str, raw_dir: str = "data/raw") -> Path:
    raw_file = Path(raw_dir) / f"raw_guardian_news_{run_date}.csv"

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    return raw_file


def preprocess_news(input_path: str | Path) -> pd.DataFrame:
    news_df = pd.read_csv(input_path)

    news_df = clean_news_data(news_df)
    news_df = add_features(news_df)
    return news_df


def save_processed_news(
    news_df: pd.DataFrame,
    run_date: str,
    output_dir: str = "data/processed",
) -> str:
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"processed_guardian_news_{run_date}.csv"

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-date",
        required=False,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Pipeline run date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_date = args.run_date

    raw_file = get_raw_file_for_date(run_date)
    print(f"Using raw file: {raw_file}")
    processed_df = preprocess_news(raw_file)
    print(f"Processed articles: {len(processed_df)}")
    output_path = save_processed_news(processed_df, run_date)
    print(f"Saved processed news to: {output_path}")


if __name__ == "__main__":
    main()