#!/bin/sh
# Container entrypoint. Migrations run inside the app under a Postgres advisory lock
# (RR_RUN_MIGRATIONS_ON_STARTUP=true), so several instances starting together are safe.
set -eu
case "${1:-serve}" in
  serve)
    shift || true
    exec researchrag serve --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  *)
    exec researchrag "$@"
    ;;
esac
