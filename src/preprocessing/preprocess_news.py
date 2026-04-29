from datetime import datetime
from pathlib import Path
import re

import pandas as pd


def get_latest_raw_file(raw_dir: str = "data/raw") -> Path:
    raw_folder = Path(raw_dir)
    raw_files = list(raw_folder.glob("raw_guardian_news_*.csv"))

    if not raw_files:
        raise FileNotFoundError("No raw Guardian news files found in data/raw.")

    latest_file = max(raw_files, key=get_file_modified_time)

    return latest_file


def get_file_modified_time(file_path: Path) -> float:
    return file_path.stat().st_mtime


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
    news_df["description_word_count"] = (
        news_df["description_clean"].str.split().str.len()
    )

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


def preprocess_news(input_path: str | Path) -> pd.DataFrame:
    news_df = pd.read_csv(input_path)

    news_df = clean_news_data(news_df)
    news_df = add_features(news_df)
    return news_df


def save_processed_news(
    news_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> str:
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = output_folder / f"processed_guardian_news_{today}.csv"

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def main() -> None:
    raw_file = get_latest_raw_file()

    print(f"Using raw file: {raw_file}")

    processed_df = preprocess_news(raw_file)

    print(f"Processed articles: {len(processed_df)}")

    output_path = save_processed_news(processed_df)

    print(f"Saved processed news to: {output_path}")


if __name__ == "__main__":
    main()