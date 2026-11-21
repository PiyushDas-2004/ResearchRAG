output "service_url" {
  description = "Cloud Run URL."
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry" {
  description = "Docker repository for images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}"
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name."
  value       = local.use_cloudsql ? google_sql_database_instance.pg[0].connection_name : ""
}

output "wif_provider" {
  description = "Workload Identity provider resource name for GitHub Actions (empty when disabled)."
  value       = var.github_repository == "" ? "" : google_iam_workload_identity_pool_provider.github[0].name
}

output "deployer_service_account" {
  description = "Service account GitHub Actions impersonates (empty when disabled)."
  value       = var.github_repository == "" ? "" : google_service_account.deployer[0].email
}
