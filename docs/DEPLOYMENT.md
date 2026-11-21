# Deployment

The project is designed to cost **nothing** to run, test, evaluate and demo:

| Path | Monthly cost | When to use |
|---|---|---|
| Local: `docker compose up` (offline components) | $0 | development, demos, CI-equivalent runs |
| GitHub Actions CI (public repository) | $0 | every push / PR |
| **Cloud Run + free-tier external Postgres (default Terraform profile)** | **$0 within free tiers** | a public demo URL |
| Cloud Run + Cloud SQL (`database_mode = "cloudsql"`) | about $10-13 (not free) | only if you accept a paid database |

Live LLM providers (OpenAI, Gemini) are paid APIs and are **off by default**; the offline
components (fake embedder, mock LLM, lexical judge) need no keys. A local OpenAI-compatible
server (Ollama) plus local embeddings is also $0 (`configs/ollama.yaml`).

> Free tiers and prices change. The figures below were checked on 2026-11-20 from the
> providers' pricing pages and third-party summaries; verify them before you deploy, and set the
> optional budget alert.

## Zero-cost cloud deployment (Cloud Run + Neon)

What makes it $0:

- **Cloud Run, request-based billing** (`cpu_idle = true`): CPU and memory are billed only while a
  request is being served, and the monthly free tier covers a demo's traffic many times over.
  `min_instances = 0` scales to zero, `max_instances = 1` caps usage. Because CPU is throttled
  between requests, uploads are ingested **inline** inside the upload request
  (`RR_INGEST_MODE=inline`, set automatically by Terraform) instead of by a background worker.
- **Database: Neon Free plan** (or another free Postgres with pgvector). Neon's Free plan currently
  includes 1 GB storage and 100 compute-hours per project per month, supports pgvector, suspends
  idle compute and does not bill overages (writes fail when the cap is reached). The synthetic
  corpus uses well under 50 MB. No Cloud SQL instance is created (`database_mode = "external"`).
- **Artifact Registry**: 0.5 GB of storage is free; the offline image is about 240 MB compressed
  and the repository's cleanup policy keeps only the newest image.
- **Secret Manager**: two secret versions (database URL, API keys) fit in the free allowance.
- **Budget alert (optional, free)**: set `billing_account` to create a 1 USD budget that e-mails
  you at the first cent of spend.

A Google Cloud billing account must be linked to the project even for free-tier usage.

### Steps

1. Create a free Neon project, copy its connection string, and turn it into a SQLAlchemy asyncpg
   URL: `postgresql+asyncpg://USER:PASSWORD@HOST/DB?ssl=require`.
2. Configure Terraform:
   ```bash
   cd infra/terraform
   cp terraform.tfvars.example terraform.tfvars   # git-ignored; fill in project, key, Neon URL
   terraform init
   terraform apply -target=google_project_service.apis -target=google_artifact_registry_repository.repo
   ```
3. Build and push the image (offline build, no provider SDKs):
   ```bash
   REGION=us-central1; PROJECT=your-gcp-project-id
   gcloud auth configure-docker $REGION-docker.pkg.dev
   docker build -t $REGION-docker.pkg.dev/$PROJECT/researchrag/api:0.1.0 .
   docker push $REGION-docker.pkg.dev/$PROJECT/researchrag/api:0.1.0
   ```
4. Apply the rest (Cloud Run, secrets, service accounts, optional WIF and budget):
   ```bash
   terraform plan    # review: no google_sql_database_instance should be planned
   terraform apply
   terraform output service_url
   ```
5. Migrations run automatically on startup (advisory-lock guarded). Load the demo corpus:
   ```bash
   URL=$(terraform output -raw service_url); KEY=your-api-key
   researchrag corpus build
   for f in data/corpus/pdfs/*.pdf; do
     curl -fsS -H "X-API-Key: $KEY" -F "file=@$f" "$URL/v1/documents" >/dev/null && echo "$f"
   done
   ```
6. Smoke test:
   ```bash
   curl -fsS "$URL/ready"
   curl -fsS -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"question":"What afternoon cooling did the CoolCrown roofs deliver in Harbourside?"}' \
     "$URL/v1/query"
   ```

### Continuous deployment (optional)

`terraform apply` with `github_repository = "PiyushDas-2004/ResearchRAG"` creates a Workload
Identity Federation pool restricted to that repository's `main` branch and a deployer service
account (`run.developer`, `artifactregistry.writer`, `actAs` on the runtime account only). Set
repository variables `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER` and `GCP_DEPLOY_SA` from
`terraform output`; then run the manual **Deploy to Cloud Run** workflow. No service-account keys
are stored in GitHub.

## Paid alternative: Cloud SQL

`database_mode = "cloudsql"` creates Cloud SQL for PostgreSQL 16 (`db-f1-micro`, 10 GB SSD,
backups) and connects through the Cloud SQL unix socket. Main cost drivers (us-central1):

| Item | Estimate |
|---|---|
| Cloud SQL `db-f1-micro` (shared core, always on) | ~$8/month |
| 10 GB SSD storage | ~$1.70/month |
| Backups (a few GB) | ~$0.25/month |
| Cloud Run (light demo, request-based) | $0 within the free tier |
| **Total** | **~$10-13/month** |

Cloud SQL is the largest cost because it runs 24/7. `cpu_always_allocated = true` (background
worker) adds instance-time billing while instances are alive. This path is **not** zero-cost and
is never the default.

## Other zero-cost options

- **Single machine you already have**: `docker compose up -d` on any machine; nothing to pay.
- **Other free Postgres providers with pgvector** (e.g. Supabase Free, which pauses inactive
  projects) work with the same `external` mode.

## Teardown

```bash
cd infra/terraform
# If you used Cloud SQL: set deletion_protection = false and apply once first.
terraform destroy
gcloud artifacts repositories delete researchrag --location=$REGION   # if not managed by Terraform state
```
Delete the Neon project from the Neon console. Terraform state contains secrets (database URL,
API keys): delete local state files or the remote state bucket when you are done.

## Status of this repository's deployment

No cloud resources have been created for this repository. Everything above is ready to apply,
but deployment was not run: it needs your Google Cloud project and Neon account, and you asked
for a zero-cost setup, so nothing is created without your explicit go-ahead.
