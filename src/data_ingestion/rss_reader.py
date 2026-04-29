from datetime import datetime, timezone

import feedparser
from bs4 import BeautifulSoup


def clean_html(raw_html: str | None) -> str | None:
    if raw_html is None:
        return None

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    return text


def get_description(entry) -> str | None:
    description = entry.get("description")

    if description is None:
        description = entry.get("summary")

    description = clean_html(description)

    return description


def normalize_entry(
    entry,
    source_name: str,
    feed_name: str,
    category: str,
) -> dict:

    article = {
        "headline": entry.get("title"),
        "description": get_description(entry),
        "url": entry.get("link"),
        "published_at": entry.get("published"),
        "source_name": source_name,
        "feed_name": feed_name,
        "category": category,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return article


def read_rss_feed(source: dict) -> list[dict]:
    source_name = source["source_name"]
    feed_name = source["feed_name"]
    category = source["category"]
    url = source["url"]

    feed = feedparser.parse(url)

    if feed.bozo:
        print(
            f"Warning: feedparser reported an issue for {feed_name}: "
            f"{feed.bozo_exception}"
        )

    articles = []

    for entry in feed.entries:
        article = normalize_entry(
            entry=entry,
            source_name=source_name,
            feed_name=feed_name,
            category=category,
        )

        articles.append(article)

    return articles