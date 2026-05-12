from pathlib import Path

from src.storage.gcs_client import GCSClient
from src.utils.config import load_yaml_config


def main():
    config = load_yaml_config("configs/gcp_config.yaml")
    bucket_name = config["gcp"]["bucket_name"]

    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    local_file = test_dir / "gcs_test.txt"
    local_file.write_text("Hello from the MLOps news project!")

    gcs = GCSClient(bucket_name=bucket_name)

    gcs.upload_file(
        local_path=str(local_file),
        gcs_path="test/gcs_test.txt",
    )

    files = gcs.list_files(prefix="test/")
    print("Files in GCS:")
    for file in files:
        print(file)


if __name__ == "__main__":
    main()