import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_topic_artifacts(model_dir: str | Path):
    model_dir = Path(model_dir)

    model_path = model_dir / "nmf_topic_model.joblib"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    metadata_path = model_dir / "topic_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, vectorizer, metadata


def predict_topics(
    input_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    model, vectorizer, metadata = load_topic_artifacts(model_dir)

    text_column = metadata["text_column"]

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in input data.")

    texts = df[text_column].fillna("").astype(str)

    tfidf_matrix = vectorizer.transform(texts)
    topic_matrix = model.transform(tfidf_matrix)

    topic_ids = topic_matrix.argmax(axis=1)
    topic_scores = topic_matrix.max(axis=1)

    topic_keywords = metadata["topic_keywords"]

    df["topic_id"] = topic_ids
    df["topic_score"] = topic_scores
    df["topic_keywords"] = [
        ", ".join(topic_keywords[str(topic_id)])
        for topic_id in topic_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved topic predictions to: {output_path}")

    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to processed input CSV.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model, vectorizer, and metadata.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save topic predictions CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predict_topics(
        input_path=args.input_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()