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

    return clean_html(description)


def get_published_at(entry) -> str | None:
    published_at = entry.get("published")

    if published_at is None:
        published_at = entry.get("updated")

    return published_at


def normalize_entry(
    entry,
    source_name: str,
    source_section: str,
    country: str,
    language: str,
) -> dict:
    article = {
        "headline": entry.get("title"),
        "description": get_description(entry),
        "url": entry.get("link"),
        "published_at": get_published_at(entry),
        "source_name": source_name,
        "source_section": source_section,
        "country": country,
        "language": language,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return article


def read_rss_feed(source: dict) -> list[dict]:
    source_name = source.get("source_name", source.get("name", "unknown"))
    source_section = source.get("source_section", source.get("category", "unknown"))
    country = source.get("country", "unknown")
    language = source.get("language", "unknown")
    url = source["url"]

    feed = feedparser.parse(url)

    if feed.bozo:
        print(
            f"Warning: feedparser reported an issue for "
            f"{source_name}/{source_section}: {feed.bozo_exception}"
        )

    articles = []

    for entry in feed.entries:
        article = normalize_entry(
            entry=entry,
            source_name=source_name,
            source_section=source_section,
            country=country,
            language=language,
        )
        articles.append(article)

    return articles