from pathlib import Path

from google.cloud import storage


class GCSClient:
    def __init__(self, bucket_name: str, project_id: str | None = None):
        self.client = storage.Client(project=project_id)
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: str, gcs_path: str) -> None:
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))

        print(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")

    def download_file(self, gcs_path: str, local_path: str) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))

        print(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")

    def list_files(self, prefix: str = "") -> list[str]:
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]