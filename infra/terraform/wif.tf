# Workload Identity Federation for GitHub Actions (no long-lived keys).
# Created only when var.github_repository is set, e.g. "PiyushDas-2004/ResearchRAG".

resource "google_iam_workload_identity_pool" "github" {
  count                     = var.github_repository == "" ? 0 : 1
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count                              = var.github_repository == "" ? 0 : 1
  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.ref == \"refs/heads/main\""
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  count        = var.github_repository == "" ? 0 : 1
  account_id   = "${var.service_name}-deploy"
  display_name = "ResearchRAG GitHub deployer"
}

resource "google_service_account_iam_member" "deployer_wif" {
  count              = var.github_repository == "" ? 0 : 1
  service_account_id = google_service_account.deployer[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github[0].name}/attribute.repository/${var.github_repository}"
}

resource "google_project_iam_member" "deployer_roles" {
  for_each = var.github_repository == "" ? toset([]) : toset(["roles/run.developer", "roles/artifactregistry.writer"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.deployer[0].email}"
}

# The deployer may deploy revisions that run as the runtime service account (and nothing else).
resource "google_service_account_iam_member" "deployer_act_as_runtime" {
  count              = var.github_repository == "" ? 0 : 1
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer[0].email}"
}
