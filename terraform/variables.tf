variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "Default GCP region"
  type        = string
  default     = "europe-west1"
}

variable "bucket_name" {
  description = "Globally unique GCS bucket name"
  type        = string
}

variable "bucket_location" {
  description = "GCS bucket location"
  type        = string
  default     = "EU"
}