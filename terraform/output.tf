output "bucket_name" {
  description = "news_mlops_bucket"
  value       = google_storage_bucket.news_mlops_bucket.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.news_mlops_bucket.name}"
}