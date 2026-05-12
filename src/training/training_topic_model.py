import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import wandb
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import load_yaml_config


WANDB_PROJECT = "guardian-news-topic-modeling"


def get_processed_files(processed_dir: str = "data/processed") -> list[Path]:
    processed_folder = Path(processed_dir)

    processed_files = []
    processed_files.extend(processed_folder.glob("live/processed_guardian_news_*.csv"))
    processed_files.extend(processed_folder.glob("backfill/processed_guardian_*.csv"))
    processed_files.extend(processed_folder.glob("api/processed_guardian_api_*.csv"))

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


def prepare_text_data(news_df: pd.DataFrame, text_column: str) -> pd.Series:
    if text_column not in news_df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    texts = news_df[text_column].fillna("").astype(str)
    texts = texts[texts.str.strip().str.len() > 0]

    if texts.empty:
        raise ValueError("No usable text data found for topic modeling.")

    return texts


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


def build_topic_model(
    n_topics: int,
    random_state: int,
    max_iter: int,
) -> NMF:
    return NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )


def get_topic_keywords(
    model: NMF,
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> dict[str, list[str]]:
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}

    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        topic_keywords[str(topic_idx)] = keywords

    return topic_keywords


def assign_topics(topic_matrix) -> tuple[list[int], list[float]]:
    topic_ids = topic_matrix.argmax(axis=1).tolist()
    topic_scores = topic_matrix.max(axis=1).tolist()

    return topic_ids, topic_scores


def save_topic_artifacts(
    model: NMF,
    vectorizer: TfidfVectorizer,
    metadata: dict,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "nmf_topic_model.joblib"
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


def save_topic_assignments(
    news_df: pd.DataFrame,
    topic_ids: list[int],
    topic_scores: list[float],
    topic_keywords: dict[str, list[str]],
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = news_df.copy()

    usable_index = result_df["topic_text"].fillna("").astype(str).str.strip().str.len() > 0
    result_df = result_df.loc[usable_index].copy()

    result_df["topic_id"] = topic_ids
    result_df["topic_score"] = topic_scores
    result_df["topic_keywords"] = [
        ", ".join(topic_keywords[str(topic_id)])
        for topic_id in topic_ids
    ]

    result_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def log_dataset_artifact(run, processed_files: list[Path]) -> None:
    artifact = wandb.Artifact(
        name="processed-guardian-news-topic-data",
        type="dataset",
        description="Processed Guardian news files used for unsupervised topic modeling.",
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
        name="guardian-nmf-topic-model",
        type="model",
        description="TF-IDF vectorizer and NMF topic model for unsupervised news topic modeling.",
    )

    artifact.add_file(model_path)
    artifact.add_file(vectorizer_path)
    artifact.add_file(metadata_path)

    run.log_artifact(artifact)


def log_topic_keywords(
    run,
    topic_keywords: dict[str, list[str]],
) -> None:
    rows = []

    for topic_id, keywords in topic_keywords.items():
        rows.append(
            {
                "topic_id": int(topic_id),
                "keywords": ", ".join(keywords),
            }
        )

    table = wandb.Table(dataframe=pd.DataFrame(rows))
    run.log({"topic_keywords": table})


def log_topic_distribution(
    run,
    topic_ids: list[int],
) -> None:
    topic_counts = (
        pd.Series(topic_ids)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    topic_counts.columns = ["topic_id", "count"]

    table = wandb.Table(dataframe=topic_counts)
    run.log({"topic_distribution": table})

    for row in topic_counts.itertuples(index=False):
        run.log({f"topic_count/topic_{row.topic_id}": row.count})


def train_topic_model(
    run_date: str,
    processed_dir: str = "data/processed",
) -> dict[str, str]:
    model_config = load_yaml_config("configs/model_config.yaml")

    text_column = model_config["text"]["text_column"]

    config = {
        "model_type": "unsupervised_topic_modeling",
        "algorithm": model_config["model"]["algorithm"],
        "text_column": text_column,
        "n_topics": model_config["model"]["n_topics"],
        "max_features": model_config["model"]["max_features"],
        "min_df": model_config["model"]["min_df"],
        "max_df": model_config["model"]["max_df"],
        "ngram_min": model_config["model"].get("ngram_min", 1),
        "ngram_max": model_config["model"].get("ngram_max", 2),
        "random_state": model_config["model"]["random_state"],
        "max_iter": model_config["model"].get("max_iter", 500),
        "top_n_keywords": model_config["model"].get("top_n_keywords", 10),
        "run_date": run_date,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"nmf-topic-model-{run_date}-{datetime.now().strftime('%H-%M-%S')}",
        config=config,
    )

    try:
        processed_files = get_processed_files(processed_dir)
        news_df = load_training_data(processed_dir)

        print(f"Loaded articles: {len(news_df)}")
        print(f"Using text column: {text_column}")

        texts = prepare_text_data(news_df, text_column)

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

        model = build_topic_model(
            n_topics=config["n_topics"],
            random_state=config["random_state"],
            max_iter=config["max_iter"],
        )

        topic_matrix = model.fit_transform(tfidf_matrix)

        topic_keywords = get_topic_keywords(
            model=model,
            vectorizer=vectorizer,
            top_n=config["top_n_keywords"],
        )

        topic_ids, topic_scores = assign_topics(topic_matrix)

        reconstruction_error = model.reconstruction_err_
        vocabulary_size = len(vectorizer.get_feature_names_out())

        print(f"Reconstruction error: {reconstruction_error:.4f}")
        print(f"Vocabulary size: {vocabulary_size}")

        print("\nDiscovered topics:")
        for topic_id, keywords in topic_keywords.items():
            print(f"Topic {topic_id}: {', '.join(keywords)}")

        run.log(
            {
                "reconstruction_error": reconstruction_error,
                "vocabulary_size": vocabulary_size,
                "num_topics": config["n_topics"],
                "mean_topic_score": sum(topic_scores) / len(topic_scores),
            }
        )

        log_topic_keywords(run, topic_keywords)
        log_topic_distribution(run, topic_ids)

        output_dir = Path("models/topic_model") / run_date
        assignments_path = Path("data/predictions") / f"topic_assignments_training_{run_date}.csv"

        metadata = {
            "run_date": run_date,
            "model_type": "unsupervised_topic_modeling",
            "algorithm": "nmf",
            "n_topics": config["n_topics"],
            "text_column": text_column,
            "topic_keywords": topic_keywords,
            "reconstruction_error": reconstruction_error,
            "vocabulary_size": vocabulary_size,
            "created_at": datetime.utcnow().isoformat(),
        }

        artifact_paths = save_topic_artifacts(
            model=model,
            vectorizer=vectorizer,
            metadata=metadata,
            output_dir=output_dir,
        )

        assignments_path = save_topic_assignments(
            news_df=news_df,
            topic_ids=topic_ids,
            topic_scores=topic_scores,
            topic_keywords=topic_keywords,
            output_path=assignments_path,
        )

        log_model_artifact(
            run=run,
            model_path=artifact_paths["model_path"],
            vectorizer_path=artifact_paths["vectorizer_path"],
            metadata_path=artifact_paths["metadata_path"],
        )

        prediction_artifact = wandb.Artifact(
            name="training-topic-assignments",
            type="predictions",
            description="Topic assignments generated on the training dataset.",
        )
        prediction_artifact.add_file(assignments_path)
        run.log_artifact(prediction_artifact)

        print(f"Saved model to: {artifact_paths['model_path']}")
        print(f"Saved vectorizer to: {artifact_paths['vectorizer_path']}")
        print(f"Saved metadata to: {artifact_paths['metadata_path']}")
        print(f"Saved training topic assignments to: {assignments_path}")

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

    train_topic_model(
        run_date=args.run_date,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()