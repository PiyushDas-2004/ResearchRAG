variable "project_id" {
  description = "GCP project ID (no default: never hardcoded)."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Cloud SQL and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "researchrag"
}

variable "image" {
  description = "Container image URI to deploy, e.g. <region>-docker.pkg.dev/<project>/researchrag/api:<tag>."
  type        = string
}

variable "api_keys" {
  description = "Comma-separated API keys accepted by the service. Stored in Secret Manager (and in Terraform state: protect your state)."
  type        = string
  sensitive   = true
}

variable "database_mode" {
  description = <<-EOT
    "external" (default, zero cost): use a free-tier managed Postgres with pgvector (e.g. Neon) via
    var.external_database_url. "cloudsql": create a Cloud SQL instance (NOT free: roughly $10+/month).
  EOT
  type        = string
  default     = "external"
  validation {
    condition     = contains(["external", "cloudsql"], var.database_mode)
    error_message = "database_mode must be \"external\" or \"cloudsql\"."
  }
}

variable "external_database_url" {
  description = "SQLAlchemy URL of an external Postgres with pgvector, e.g. postgresql+asyncpg://user:pass@host/db?ssl=require (database_mode = external)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cpu_always_allocated" {
  description = "false (default, zero cost) = request-based billing, uploads are ingested inline. true = instance-based billing with a background worker (costs money while instances are alive)."
  type        = bool
  default     = false
}

variable "billing_account" {
  description = "Optional billing account ID; when set, a budget with a 1 USD alert threshold is created as a safety net."
  type        = string
  default     = ""
}

variable "db_tier" {
  description = "Cloud SQL machine tier. db-f1-micro is the cheapest shared-core option; use db-custom-1-3840 or larger for real workloads."
  type        = string
  default     = "db-f1-micro"
}

variable "db_disk_gb" {
  description = "Cloud SQL disk size in GB."
  type        = number
  default     = 10
}

variable "deletion_protection" {
  description = "Protect the Cloud SQL instance from deletion. Set false before `terraform destroy`."
  type        = bool
  default     = true
}

variable "min_instances" {
  description = "Minimum Cloud Run instances (0 = scale to zero to control cost)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum Cloud Run instances (1 keeps usage predictable and inside the free tier)."
  type        = number
  default     = 1
}

variable "concurrency" {
  description = "Concurrent requests per instance (streaming responses hold a request open)."
  type        = number
  default     = 20
}

variable "request_timeout_s" {
  description = "Cloud Run request timeout in seconds (long enough for streamed answers)."
  type        = number
  default     = 300
}

variable "pipeline_config" {
  description = "Pipeline config file inside the image (configs/default.yaml is fully offline)."
  type        = string
  default     = "/app/configs/default.yaml"
}

variable "allow_unauthenticated" {
  description = "Allow public (IAM-unauthenticated) invocations. The API still requires an API key."
  type        = bool
  default     = true
}

variable "github_repository" {
  description = "owner/repo allowed to deploy via Workload Identity Federation. Empty disables WIF resources."
  type        = string
  default     = ""
}
