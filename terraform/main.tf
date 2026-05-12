terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "news_mlops_bucket" {
  name          = var.bucket_name
  location      = var.bucket_location
  storage_class = "STANDARD"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  force_destroy = true

  labels = {
    project = "news-mlops"
    env     = "dev"
  }
}