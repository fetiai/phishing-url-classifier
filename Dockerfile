# syntax=docker/dockerfile:1

# The artifact bundle is BAKED IN, never volume-mounted.
#
# A mounted artifacts directory is the most common way a service ends up serving a model
# nobody can identify: the image tag says one thing, the mount says another, and the only
# way to find out which models answered a given request is to go and look at the running
# host. Baking makes the image tag a complete description of behaviour, and makes rollback
# "deploy the previous tag" rather than an archaeology exercise.
#
# Train in CI, publish the bundle as a release asset, and COPY it in at build time.

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder
# 3.13 matches the interpreter the tests and the type checker run against. Leaving a
# version gap between development and production means the pinned wheels are resolved
# twice and the numbers only have to agree by luck.

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # `streamlit run app/Home.py` puts the *script's* directory on sys.path, not the
    # working directory, so `from app.state import ...` finds nothing. Locally an editable
    # install hides this by putting the repo root on the path; the image installs only the
    # pinned requirements, so it has to be stated.
    PYTHONPATH=/app \
    # BLAS threads are capped deliberately. The KNN distance kernel will happily saturate
    # every core it can see, and on a 2 vCPU box that starves the web server that has to
    # answer the request.
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    ARTIFACTS_DIR=/app/artifacts/v1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 10001 phishguard

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY phishguard/ ./phishguard/
COPY app/ ./app/
COPY pyproject.toml README.md ./

# The bundle. Everything the service will ever load.
COPY artifacts/ ./artifacts/

RUN chown -R phishguard:phishguard /app
USER phishguard

# Fail the build if the shipped bundle does not reproduce its own golden row. Catching it
# here means a broken bundle never reaches a registry, let alone a host.
RUN python -m phishguard.selftest --golden

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app/Home.py"]
