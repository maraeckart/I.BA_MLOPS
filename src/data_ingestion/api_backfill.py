import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# - load API key
# - load configured Guardian sections
# - request articles for each source_section
# - normalize API results
# - remove duplicates
# - save CSV file

# python -m src.data_ingestion.guardian_api_backfill \
#   --from-date 2026-02-01 \
#   --to-date 2026-04-29


API_BASE_URL = "https://content.guardianapis.com/search"

def load_sources(config_path: str = "configs/sources.yaml") -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["sources"]

def get_api_key() -> str:
    load_dotenv()

    api_key = os.getenv("GUARDIAN_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GUARDIAN_API_KEY. Add it to your .env file or environment."
        )

    return api_key


def clean_html(raw_html: str | None) -> str:
    if raw_html is None:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    return text


def build_request_params(
    source: dict,
    from_date: str,
    to_date: str,
    page: int,
    page_size: int,
    api_key: str,
) -> dict:

    params = {
        "api-key": api_key,
        "section": source["api_section"],
        "from-date": from_date,
        "to-date": to_date,
        "page": page,
        "page-size": page_size,
        "order-by": "newest",
        "show-fields": "headline,trailText,shortUrl,bodyText",
    }
    return params

def request_guardian_page(params: dict) -> dict:
    response = requests.get(API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("response", {}).get("status") != "ok":
        raise ValueError(f"Guardian API returned unexpected response: {data}")

    return data


def extract_description(fields: dict) -> str:
    trail_text = fields.get("trailText")

    if trail_text:
        return clean_html(trail_text)

    body_text = fields.get("bodyText")

    if body_text:
        return clean_html(body_text)

    return ""


def normalize_api_result(result: dict, source: dict) -> dict:
    fields = result.get("fields", {})
    headline = fields.get("headline")

    if not headline:
        headline = result.get("webTitle")

    article = {
        "headline": headline,
        "description": extract_description(fields),
        "url": result.get("webUrl"),
        "published_at": result.get("webPublicationDate"),
        "source_name": source["source_name"],
        "source_section": source["source_section"],
        "country": source.get("country"),
        "language": source.get("language"),
        "ingestion_timestamp": datetime.utcnow().isoformat(),
    }

    return article


def collect_source_articles(
    source: dict,
    from_date: str,
    to_date: str,
    api_key: str,
    page_size: int,
    sleep_seconds: float,
    max_pages: int | None,
) -> list[dict]:
    source_section = source["source_section"]
    api_section = source["api_section"]

    print(f"Collecting {source_section} from API section '{api_section}'...")

    articles = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        if max_pages is not None and page > max_pages:
            print(f"Reached max_pages={max_pages} for {source_section}.")
            break

        params = build_request_params(
            source=source,
            from_date=from_date,
            to_date=to_date,
            page=page,
            page_size=page_size,
            api_key=api_key,
        )

        data = request_guardian_page(params)
        response_data = data["response"]

        total_pages = response_data.get("pages", 1)
        results = response_data.get("results", [])

        print(
            f"{source_section}: page {page}/{total_pages}, "
            f"articles on page: {len(results)}"
        )

        for result in results:
            article = normalize_api_result(result, source)
            articles.append(article)

        page = page + 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print(f"Collected {len(articles)} articles for {source_section}.")

    return articles


def collect_all_articles(
    from_date: str,
    to_date: str,
    config_path: str,
    page_size: int,
    sleep_seconds: float,
    max_pages: int | None,
) -> pd.DataFrame:
    api_key = get_api_key()
    sources = load_sources(config_path)
    all_articles = []

    for source in sources:
        if "api_section" not in source:
            continue

        articles = collect_source_articles(
            source=source,
            from_date=from_date,
            to_date=to_date,
            api_key=api_key,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
            max_pages=max_pages,
        )

        all_articles.extend(articles)

    news_df = pd.DataFrame(all_articles)

    if news_df.empty:
        return news_df

    news_df = news_df.drop_duplicates(
        subset=["headline", "url"],
        keep="first",
    )

    return news_df


def save_api_data(
    news_df: pd.DataFrame,
    from_date: str,
    to_date: str,
    output_dir: str = "data/raw/api",
) -> str:

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"raw_guardian_api_{from_date}_{to_date}.csv"

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--to-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--config-path",
        default="configs/sources.yaml",
        help="Path to sources YAML file.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of results per API page.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Pause between API requests.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum pages per source_section for testing.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    news_df = collect_all_articles(
        from_date=args.from_date,
        to_date=args.to_date,
        config_path=args.config_path,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
        max_pages=args.max_pages,
    )

    print(f"Total collected articles after duplicate removal: {len(news_df)}")

    if news_df.empty:
        print("No articles collected. No file saved.")
        return

    output_path = save_api_data(
        news_df=news_df,
        from_date=args.from_date,
        to_date=args.to_date,
    )

    print(f"Saved Guardian API backfill data to: {output_path}")


if __name__ == "__main__":
    main()