import argparse
from pathlib import Path

from src.storage.gcs_client import GCSClient
from src.utils.config import load_yaml_config


def upload_file_to_gcs(
    local_path: str | Path,
    gcs_prefix: str,
    config_path: str = "configs/gcp_config.yaml",
) -> str:
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    config = load_yaml_config(config_path)
    project_id = config["gcp"]["project_id"]
    bucket_name = config["gcp"]["bucket_name"]
    

    gcs_path = f"{gcs_prefix}/{local_path.name}"

    gcs = GCSClient(bucket_name=bucket_name,project_id=project_id)
    gcs.upload_file(
        local_path=str(local_path),
        gcs_path=gcs_path,
    )

    full_gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"Uploaded file to: {full_gcs_uri}")

    return full_gcs_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local-path",
        required=True,
        help="Local file path to upload.",
    )

    parser.add_argument(
        "--gcs-prefix",
        required=True,
        help="GCS prefix/folder, for example raw/rss/live.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    upload_file_to_gcs(
        local_path=args.local_path,
        gcs_prefix=args.gcs_prefix,
    )


if __name__ == "__main__":
    main()