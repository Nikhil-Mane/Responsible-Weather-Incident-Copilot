# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are installed in a builder, the runtime image is slim and non-root.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui,agents,observability]"

FROM python:3.11-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    CREWAI_TRACING_ENABLED=false \
    COPILOT_ENV=prod \
    COPILOT_STATE_DIR=/app/var \
    COPILOT_GENERATED_DIR=/app/var/generated
RUN useradd --create-home --uid 10001 copilot
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app.py ./
COPY scripts ./scripts
COPY eval ./eval
COPY data ./data
RUN mkdir -p /app/var && chown -R copilot:copilot /app
USER copilot
EXPOSE 8501
VOLUME ["/app/var"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"
# COPILOT_APPROVAL_SECRET (32+ characters) must be provided at run time; there is no default.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
