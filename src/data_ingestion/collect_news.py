import argparse
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


def get_source_label(source: dict) -> str:
    source_name = source.get("source_name", source.get("name", "unknown"))
    source_section = source.get("source_section", source.get("category", "unknown"))

    return f"{source_name}/{source_section}"


def collect_source_news(source: dict) -> list[dict]:
    source_label = get_source_label(source)

    print(f"Collecting articles from {source_label}...")

    try:
        articles = read_rss_feed(source)
        print(f"Collected {len(articles)} articles from {source_label}.")
        return articles

    except Exception as error:
        print(f"Failed to collect articles from {source_label}: {error}")
        return []


def remove_duplicates(news_df: pd.DataFrame) -> pd.DataFrame:
    if not {"headline", "url"}.issubset(news_df.columns):
        return news_df

    news_df = news_df.drop_duplicates(
        subset=["headline", "url"],
        keep="first",
    )
    return news_df


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


def save_raw_news(
    news_df: pd.DataFrame,
    run_date: str,
    output_dir: str = "data/raw/rss/live",
) -> str:
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"raw_news_{run_date}.csv"

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

    news_df = collect_all_news()
    print(f"Total collected articles after duplicate removal: {len(news_df)}")

    if news_df.empty:
        print("No articles collected. No file saved.")
        return

    output_path = save_raw_news(
        news_df=news_df,
        run_date=run_date,
        output_dir="data/raw/rss/live",
    )

    print(f"Saved raw news to: {output_path}")


if __name__ == "__main__":
    main()