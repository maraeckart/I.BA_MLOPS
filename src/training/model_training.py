from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


WANDB_PROJECT = "guardian-news-topic-classification"


def get_processed_files(processed_dir: str = "data/processed") -> list[Path]:
    processed_folder = Path(processed_dir)

    processed_files = []
    processed_files.extend(processed_folder.glob("processed_guardian_news_*.csv"))
    processed_files.extend(processed_folder.glob("api/processed_guardian_api_*.csv"))

    if not processed_files:
        raise FileNotFoundError("No processed Guardian news files found.")
    return processed_files

def load_training_data(processed_dir: str = "data/processed") -> pd.DataFrame:
    processed_files = get_processed_files(processed_dir)
    dataframes = []

    for file_path in processed_files:
        news_df = pd.read_csv(file_path)
        dataframes.append(news_df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df = combined_df.drop_duplicates(
        subset=["headline", "url"],
        keep="first",
    )
    return combined_df


def get_target_column(news_df: pd.DataFrame) -> str:
    if "category" in news_df.columns:
        return "category"

    raise ValueError("No target column found. Expected topic_category or category.")


def prepare_training_data(news_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    target_column = get_target_column(news_df)

    required_columns = ["headline_clean", target_column]

    for column in required_columns:
        if column not in news_df.columns:
            raise ValueError(f"Missing required column: {column}")

    news_df = news_df.dropna(subset=["headline_clean", target_column])
    news_df = news_df[news_df["headline_clean"].str.len() > 0]

    X = news_df["headline_clean"]
    y = news_df[target_column]

    return X, y


def build_model(
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    max_features=max_features,
                    ngram_range=(ngram_min, ngram_max),
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }
    return metrics


def save_model(model: Pipeline, output_dir: str = "models") -> str:
    model_folder = Path(output_dir)
    model_folder.mkdir(parents=True, exist_ok=True)

    output_path = model_folder / "topic_classifier_pipeline.joblib"

    joblib.dump(model, output_path)

    return str(output_path)


def save_metrics(metrics: dict, output_dir: str = "models") -> str:
    model_folder = Path(output_dir)
    model_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = model_folder / f"training_metrics_{timestamp}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Baseline Topic Classification Model\n")
        file.write("=" * 40)
        file.write("\n\n")

        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        file.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")

        file.write("\nClassification Report:\n")
        file.write(metrics["classification_report"])

    return str(output_path)


def log_dataset_artifact(
    run,
    processed_files: list[Path],
) -> None:
    artifact = wandb.Artifact(
        name="processed-guardian-news",
        type="dataset",
        description="Processed Guardian news CSV files used for training.",
    )

    for file_path in processed_files:
        artifact.add_file(str(file_path))
    run.log_artifact(artifact)


def log_model_artifact(
    run,
    model_path: str,
    metrics_path: str,
) -> None:
    artifact = wandb.Artifact(
        name="guardian-topic-classifier",
        type="model",
        description="TF-IDF and Logistic Regression baseline topic classifier.",
    )

    artifact.add_file(model_path)
    artifact.add_file(metrics_path)
    run.log_artifact(artifact)


def log_category_distribution(
    run,
    news_df: pd.DataFrame,
    target_column: str,
) -> None:

    category_counts = news_df[target_column].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    table = wandb.Table(dataframe=category_counts)

    run.log({"category_distribution": table})

    for row in category_counts.itertuples(index=False):
        metric_name = f"category_count/{row.category}"
        run.log({metric_name: row.count})


def train_model() -> None:
    config = {
        "model_type": "tfidf_logistic_regression",
        "text_column": "headline_clean",
        "test_size": 0.2,
        "random_state": 42,
        "max_features": 5000,
        "ngram_min": 1,
        "ngram_max": 2,
        "min_df": 1,
        "class_weight": "balanced",
        "max_iter": 1000,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"baseline-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        config=config,
    )

    try:
        processed_files = get_processed_files()
        news_df = load_training_data()

        print(f"Loaded articles: {len(news_df)}")

        target_column = get_target_column(news_df)
        print(f"Using target column: {target_column}")

        print("Category distribution:")
        print(news_df[target_column].value_counts())

        X, y = prepare_training_data(news_df)

        run.config.update(
            {
                "target_column": target_column,
                "total_articles_before_cleaning": len(news_df),
                "training_rows_after_cleaning": len(X),
                "num_classes": y.nunique(),
            }
        )

        log_category_distribution(run, news_df, target_column)
        log_dataset_artifact(run, processed_files)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            stratify=y,
            random_state=config["random_state"],
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        model = build_model(
            max_features=config["max_features"],
            ngram_min=config["ngram_min"],
            ngram_max=config["ngram_max"],
            min_df=config["min_df"],
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        print("\nModel evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\nClassification report:")
        print(metrics["classification_report"])

        run.log(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "training_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        model_path = save_model(model)
        metrics_path = save_metrics(metrics)

        log_model_artifact(
            run=run,
            model_path=model_path,
            metrics_path=metrics_path,
        )
        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")

    finally:
        run.finish()


if __name__ == "__main__":
    train_model()