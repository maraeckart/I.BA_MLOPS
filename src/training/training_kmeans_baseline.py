import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import wandb
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import load_yaml_config


WANDB_PROJECT = "news-topic-modeling"


def get_processed_files(processed_dir: str = "data/processed") -> list[Path]:
    processed_folder = Path(processed_dir)

    processed_files = []
    processed_files.extend(processed_folder.glob("live/processed_news_*.csv"))
    processed_files.extend(processed_folder.glob("backfill/processed_*.csv"))
    processed_files.extend(processed_folder.glob("api/processed_api_*.csv"))

    if not processed_files:
        raise FileNotFoundError(f"No processed files found in {processed_folder}")

    return processed_files


def load_training_data(processed_dir: str = "data/processed") -> pd.DataFrame:
    processed_files = get_processed_files(processed_dir)
    dataframes = []

    for file_path in processed_files:
        news_df = pd.read_csv(file_path)
        news_df["source_file"] = str(file_path)
        dataframes.append(news_df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    if {"headline", "url"}.issubset(combined_df.columns):
        combined_df = combined_df.drop_duplicates(
            subset=["headline", "url"],
            keep="first",
        )
    elif "headline" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(
            subset=["headline"],
            keep="first",
        )

    return combined_df


def prepare_text_data(news_df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if text_column not in news_df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    usable_index = news_df[text_column].fillna("").astype(str).str.strip().str.len() > 0
    filtered_df = news_df.loc[usable_index].copy()
    texts = filtered_df[text_column].fillna("").astype(str)

    if texts.empty:
        raise ValueError("No usable text data found for KMeans baseline.")

    return filtered_df, texts


def build_vectorizer(
    max_features: int,
    min_df: int,
    max_df: float,
    ngram_min: int,
    ngram_max: int,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
        stop_words="english",
    )


def build_kmeans_model(
    n_topics: int,
    random_state: int,
    max_iter: int,
) -> KMeans:
    return KMeans(
        n_clusters=n_topics,
        random_state=random_state,
        max_iter=max_iter,
        n_init="auto",
    )


def get_cluster_keywords(
    model: KMeans,
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> dict[str, list[str]]:
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_idx, cluster_weights in enumerate(model.cluster_centers_):
        top_indices = cluster_weights.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[str(cluster_idx)] = keywords

    return cluster_keywords


def assign_clusters(model: KMeans, tfidf_matrix) -> tuple[list[int], list[float]]:
    cluster_ids = model.predict(tfidf_matrix).tolist()

    # KMeans does not produce topic probabilities.
    # This score is a simple distance-based confidence proxy:
    # smaller distance = closer to assigned cluster center.
    distances = model.transform(tfidf_matrix)
    min_distances = distances.min(axis=1)

    max_distance = min_distances.max()

    if max_distance == 0:
        cluster_scores = [1.0 for _ in min_distances]
    else:
        cluster_scores = (1 - (min_distances / max_distance)).tolist()

    return cluster_ids, cluster_scores


def save_kmeans_artifacts(
    model: KMeans,
    vectorizer: TfidfVectorizer,
    metadata: dict,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "kmeans_topic_model.joblib"
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    metadata_path = output_dir / "topic_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
        "metadata_path": str(metadata_path),
    }


def save_cluster_assignments(
    news_df: pd.DataFrame,
    cluster_ids: list[int],
    cluster_scores: list[float],
    cluster_keywords: dict[str, list[str]],
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = news_df.copy()

    result_df["topic_id"] = cluster_ids
    result_df["topic_score"] = cluster_scores
    result_df["topic_keywords"] = [
        ", ".join(cluster_keywords[str(cluster_id)])
        for cluster_id in cluster_ids
    ]

    result_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def log_dataset_artifact(run, processed_files: list[Path]) -> None:
    artifact = wandb.Artifact(
        name="processed-news-topic-data-kmeans-baseline",
        type="dataset",
        description="Processed news files used for KMeans topic baseline.",
    )

    for file_path in processed_files:
        artifact.add_file(str(file_path))

    run.log_artifact(artifact)


def log_model_artifact(
    run,
    model_path: str,
    vectorizer_path: str,
    metadata_path: str,
) -> None:
    artifact = wandb.Artifact(
        name="kmeans-topic-baseline",
        type="model",
        description="TF-IDF vectorizer and KMeans clustering baseline for news topic modeling.",
    )

    artifact.add_file(model_path)
    artifact.add_file(vectorizer_path)
    artifact.add_file(metadata_path)

    run.log_artifact(artifact)


def log_cluster_keywords(
    run,
    cluster_keywords: dict[str, list[str]],
) -> None:
    rows = []

    for topic_id, keywords in cluster_keywords.items():
        rows.append(
            {
                "topic_id": int(topic_id),
                "keywords": ", ".join(keywords),
            }
        )

    table = wandb.Table(dataframe=pd.DataFrame(rows))
    run.log({"topic_keywords": table})


def log_cluster_distribution(
    run,
    cluster_ids: list[int],
) -> None:
    topic_counts = (
        pd.Series(cluster_ids)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    topic_counts.columns = ["topic_id", "count"]

    table = wandb.Table(dataframe=topic_counts)
    run.log({"topic_distribution": table})

    for row in topic_counts.itertuples(index=False):
        run.log({f"topic_count/topic_{row.topic_id}": row.count})


def train_kmeans_baseline(
    run_date: str,
    processed_dir: str = "data/processed",
) -> dict[str, str]:
    model_config = load_yaml_config("configs/model_config.yaml")

    text_column = model_config["text"]["text_column"]

    config = {
        "model_type": "unsupervised_topic_modeling_baseline",
        "algorithm": "kmeans",
        "text_column": text_column,
        "n_topics": model_config["model"]["n_topics"],
        "max_features": model_config["model"]["max_features"],
        "min_df": model_config["model"]["min_df"],
        "max_df": model_config["model"]["max_df"],
        "ngram_min": model_config["model"].get("ngram_min", 1),
        "ngram_max": model_config["model"].get("ngram_max", 2),
        "random_state": model_config["model"]["random_state"],
        "max_iter": model_config["model"].get("max_iter", 300),
        "top_n_keywords": model_config["model"].get("top_n_keywords", 10),
        "run_date": run_date,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"kmeans-baseline-{run_date}-{datetime.now().strftime('%H-%M-%S')}",
        config=config,
    )

    try:
        processed_files = get_processed_files(processed_dir)
        news_df = load_training_data(processed_dir)

        print(f"Loaded articles: {len(news_df)}")
        print(f"Using text column: {text_column}")

        filtered_df, texts = prepare_text_data(news_df, text_column)

        run.config.update(
            {
                "num_processed_files": len(processed_files),
                "total_articles_before_text_filtering": len(news_df),
                "training_rows_after_text_filtering": len(texts),
            }
        )

        log_dataset_artifact(run, processed_files)

        vectorizer = build_vectorizer(
            max_features=config["max_features"],
            min_df=config["min_df"],
            max_df=config["max_df"],
            ngram_min=config["ngram_min"],
            ngram_max=config["ngram_max"],
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        model = build_kmeans_model(
            n_topics=config["n_topics"],
            random_state=config["random_state"],
            max_iter=config["max_iter"],
        )

        model.fit(tfidf_matrix)

        cluster_keywords = get_cluster_keywords(
            model=model,
            vectorizer=vectorizer,
            top_n=config["top_n_keywords"],
        )

        cluster_ids, cluster_scores = assign_clusters(model, tfidf_matrix)

        vocabulary_size = len(vectorizer.get_feature_names_out())
        inertia = model.inertia_
        mean_cluster_score = sum(cluster_scores) / len(cluster_scores)

        print(f"KMeans inertia: {inertia:.4f}")
        print(f"Vocabulary size: {vocabulary_size}")

        print("\nDiscovered KMeans baseline topics:")
        for topic_id, keywords in cluster_keywords.items():
            print(f"Topic {topic_id}: {', '.join(keywords)}")

        run.log(
            {
                "kmeans_inertia": inertia,
                "vocabulary_size": vocabulary_size,
                "num_topics": config["n_topics"],
                "mean_topic_score": mean_cluster_score,
            }
        )

        log_cluster_keywords(run, cluster_keywords)
        log_cluster_distribution(run, cluster_ids)

        output_dir = Path("models/topic_model_kmeans_baseline") / run_date
        assignments_path = (
            Path("data/predictions")
            / f"kmeans_topic_assignments_training_{run_date}.csv"
        )

        metadata = {
            "run_date": run_date,
            "model_type": "unsupervised_topic_modeling_baseline",
            "algorithm": "kmeans",
            "n_topics": config["n_topics"],
            "text_column": text_column,
            "topic_keywords": cluster_keywords,
            "kmeans_inertia": inertia,
            "vocabulary_size": vocabulary_size,
            "created_at": datetime.utcnow().isoformat(),
        }

        artifact_paths = save_kmeans_artifacts(
            model=model,
            vectorizer=vectorizer,
            metadata=metadata,
            output_dir=output_dir,
        )

        assignments_path = save_cluster_assignments(
            news_df=filtered_df,
            cluster_ids=cluster_ids,
            cluster_scores=cluster_scores,
            cluster_keywords=cluster_keywords,
            output_path=assignments_path,
        )

        log_model_artifact(
            run=run,
            model_path=artifact_paths["model_path"],
            vectorizer_path=artifact_paths["vectorizer_path"],
            metadata_path=artifact_paths["metadata_path"],
        )

        prediction_artifact = wandb.Artifact(
            name="kmeans-training-topic-assignments",
            type="predictions",
            description="KMeans topic assignments generated on the training dataset.",
        )
        prediction_artifact.add_file(assignments_path)
        run.log_artifact(prediction_artifact)

        print(f"Saved KMeans model to: {artifact_paths['model_path']}")
        print(f"Saved vectorizer to: {artifact_paths['vectorizer_path']}")
        print(f"Saved metadata to: {artifact_paths['metadata_path']}")
        print(f"Saved KMeans training topic assignments to: {assignments_path}")

        return {
            **artifact_paths,
            "assignments_path": assignments_path,
        }

    finally:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-date",
        required=False,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Training run date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--processed-dir",
        required=False,
        default="data/processed",
        help="Directory containing processed CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_kmeans_baseline(
        run_date=args.run_date,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()