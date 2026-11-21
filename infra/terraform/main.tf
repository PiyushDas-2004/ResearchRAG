locals {
  use_cloudsql = var.database_mode == "cloudsql"
  services = concat([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ], local.use_cloudsql ? ["sqladmin.googleapis.com"] : [], var.billing_account == "" ? [] : ["billingbudgets.googleapis.com"])
  db_name = "researchrag"
  db_user = "researchrag"
}

resource "google_project_service" "apis" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------- registry
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "researchrag"
  format        = "DOCKER"
  description   = "ResearchRAG container images"

  # Keep only the newest image so storage stays inside the 0.5 GB free tier.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-latest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 1
    }
  }
  cleanup_policies {
    id     = "delete-older"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }
  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------- runtime identity (least privilege)
resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "ResearchRAG Cloud Run runtime"
}

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset(concat(
    ["roles/logging.logWriter", "roles/monitoring.metricWriter"],
    local.use_cloudsql ? ["roles/cloudsql.client"] : [],
  ))
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# ---------------------------------------------------------------- Cloud SQL (optional, NOT free)
resource "random_password" "db" {
  count   = local.use_cloudsql ? 1 : 0
  length  = 32
  special = false
}

resource "google_sql_database_instance" "pg" {
  count               = local.use_cloudsql ? 1 : 0
  name                = "${var.service_name}-pg"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.db_tier
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"
    disk_size         = var.db_disk_gb
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }

    ip_configuration {
      # Public IP with no authorized networks: only the Cloud SQL connector (IAM-authorised)
      # can reach it. Switch to private IP + Serverless VPC access for stricter setups.
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    database_flags {
      name  = "max_connections"
      value = "50"
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "db" {
  count    = local.use_cloudsql ? 1 : 0
  name     = local.db_name
  instance = google_sql_database_instance.pg[0].name
}

resource "google_sql_user" "app" {
  count    = local.use_cloudsql ? 1 : 0
  name     = local.db_user
  instance = google_sql_database_instance.pg[0].name
  password = random_password.db[0].result
}

# ---------------------------------------------------------------- secrets
resource "google_secret_manager_secret" "database_url" {
  secret_id = "${var.service_name}-database-url"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret = google_secret_manager_secret.database_url.id
  # Cloud SQL: asyncpg connects through the unix socket mounted at /cloudsql.
  secret_data = local.use_cloudsql ? "postgresql+asyncpg://${local.db_user}:${random_password.db[0].result}@/${local.db_name}?host=/cloudsql/${google_sql_database_instance.pg[0].connection_name}" : var.external_database_url
}

resource "google_secret_manager_secret" "api_keys" {
  secret_id = "${var.service_name}-api-keys"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "api_keys" {
  secret      = google_secret_manager_secret.api_keys.id
  secret_data = var.api_keys
}

# Optional provider keys: create the secrets empty; add versions yourself with
#   printf '%s' "$OPENAI_API_KEY" | gcloud secrets versions add researchrag-openai-api-key --data-file=-
resource "google_secret_manager_secret" "provider_keys" {
  for_each  = toset(["openai-api-key", "gemini-api-key"])
  secret_id = "${var.service_name}-${each.value}"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "runtime_access" {
  for_each = merge(
    {
      database_url = google_secret_manager_secret.database_url.id
      api_keys     = google_secret_manager_secret.api_keys.id
    },
    { for k, s in google_secret_manager_secret.provider_keys : k => s.id },
  )
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# ---------------------------------------------------------------- Cloud Run
resource "google_cloud_run_v2_service" "api" {
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "${var.request_timeout_s}s"
    max_instance_request_concurrency = var.concurrency

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    dynamic "volumes" {
      for_each = local.use_cloudsql ? [1] : []
      content {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.pg[0].connection_name]
        }
      }
    }

    containers {
      image = var.image
      ports {
        container_port = 8000
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        # cpu_idle = true is request-based billing (free-tier friendly): CPU only while serving a
        # request, so uploads are ingested inline. false keeps CPU for the background worker but is
        # billed for the whole instance lifetime.
        cpu_idle          = !var.cpu_always_allocated
        startup_cpu_boost = false
      }
      dynamic "volume_mounts" {
        for_each = local.use_cloudsql ? [1] : []
        content {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
      env {
        name  = "RR_INGEST_MODE"
        value = var.cpu_always_allocated ? "background" : "inline"
      }
      env {
        name  = "RR_RUN_WORKER"
        value = var.cpu_always_allocated ? "true" : "false"
      }
      env {
        name  = "RR_ENVIRONMENT"
        value = "prod"
      }
      env {
        name  = "RR_RUN_MIGRATIONS_ON_STARTUP"
        value = "true"
      }
      env {
        name  = "RR_PIPELINE_CONFIG"
        value = var.pipeline_config
      }
      env {
        name  = "RR_DB_POOL_SIZE"
        value = "5"
      }
      env {
        name  = "RR_PARSE_MEMORY_LIMIT_MB"
        value = "1024"
      }
      env {
        name = "RR_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "RR_API_KEYS"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_keys.secret_id
            version = "latest"
          }
        }
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 2
        period_seconds        = 3
        failure_threshold     = 20
      }
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime_access,
    google_project_iam_member.runtime_roles,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_version.api_keys,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------- optional cost safety net
resource "google_billing_budget" "guard" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "${var.service_name} zero-cost guard"
  budget_filter {
    projects = ["projects/${var.project_id}"]
  }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = "1"
    }
  }
  threshold_rules {
    threshold_percent = 0.01
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
