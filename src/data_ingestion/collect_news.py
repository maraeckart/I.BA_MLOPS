from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data_ingestion.rss_reader import read_rss_feed


def load_sources(config_path: str = "configs/sources.yaml") -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    sources = config["sources"]

    return sources


def collect_source_news(source: dict) -> list[dict]:
    feed_name = source["feed_name"]

    print(f"Collecting articles from {feed_name}...")

    try:
        articles = read_rss_feed(source)
        print(f"Collected {len(articles)} articles from {feed_name}.")

        return articles

    except Exception as error:
        print(f"Failed to collect articles from {feed_name}: {error}")

        return []


def collect_all_news(config_path: str = "configs/sources.yaml") -> pd.DataFrame:
    sources = load_sources(config_path)
    all_articles = []

    for source in sources:
        articles = collect_source_news(source)
        all_articles.extend(articles)

    news_df = pd.DataFrame(all_articles)

    if news_df.empty:
        return news_df

    news_df = remove_duplicates(news_df)

    return news_df


def remove_duplicates(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.drop_duplicates(
        subset=["headline", "url"],
        keep="first",
    )
    return news_df


def save_raw_news(news_df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = output_folder / f"raw_guardian_news_{today}.csv"

    news_df.to_csv(output_path, index=False, encoding="utf-8")
    return str(output_path)


def main() -> None:
    news_df = collect_all_news()

    print(f"Total collected articles after duplicate removal: {len(news_df)}")

    if news_df.empty:
        print("No articles collected. No file saved.")
        return

    output_path = save_raw_news(news_df)

    print(f"Saved raw news to: {output_path}")


if __name__ == "__main__":
    main()