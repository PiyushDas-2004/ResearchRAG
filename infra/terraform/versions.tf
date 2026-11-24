terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.13"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6"
    }
  }
  # Configure a remote backend (e.g. a GCS bucket) before using this for anything shared:
  # terraform { backend "gcs" { bucket = "<your-state-bucket>" prefix = "researchrag" } }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
