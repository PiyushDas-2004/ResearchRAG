# syntax=docker/dockerfile:1.7
# ---------- build stage: build a wheel and install it into a venv ----------
FROM python:3.12-slim AS build
# Provider SDK extras are opt-in so the default offline image stays small (registry storage is
# part of the zero-cost budget). Example: --build-arg EXTRAS=openai,gemini,tiktoken
ARG EXTRAS=""
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Behind a TLS-intercepting proxy, pass its CA as a build secret (never baked into the image):
#   docker build --secret id=pipca,src=/path/to/ca.crt --build-arg HTTPS_PROXY=... .
RUN --mount=type=secret,id=pipca,required=false \
    if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi \
 && pip install --upgrade pip build \
 && python -m build --wheel --outdir /dist \
 && WHEEL="/dist/$(ls /dist | grep '\.whl$')" \
 && if [ -n "$EXTRAS" ]; then pip install "${WHEEL}[${EXTRAS}]"; else pip install "$WHEEL"; fi \
 && pip install --upgrade "setuptools>=78.1.1"  # CVE-2025-47273 (flagged by Trivy in CI)

# ---------- runtime stage: slim, non-root ----------
FROM python:3.12-slim AS runtime
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="ResearchRAG" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/PiyushDas-2004/ResearchRAG" \
      org.opencontainers.image.licenses="MIT"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    RR_PIPELINE_CONFIG=/app/configs/default.yaml \
    RR_PRICING_FILE=/app/configs/pricing.yaml
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app data ./data
COPY --chown=app:app docker/entrypoint.sh ./entrypoint.sh
RUN chmod 0555 ./entrypoint.sh
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve"]
