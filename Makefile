PY ?= .venv/bin/python
PIP ?= $(PY) -m pip
IMAGE ?= phishguard:local

.PHONY: help install lint fmt type test test-fast test-network train train-legacy \
        fixtures agreement app docker-build docker-test selftest clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install runtime + dev dependencies
	test -d .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e . --no-deps

lint: ## ruff check
	$(PY) -m ruff check phishguard app scripts tests

fmt: ## ruff format + autofix
	$(PY) -m ruff check --fix phishguard app scripts tests
	$(PY) -m ruff format phishguard app scripts tests

type: ## mypy
	$(PY) -m mypy phishguard

test: ## Full offline suite (excludes network/container tiers)
	$(PY) -m pytest -m "not network and not container"

test-fast: ## Tiers that run in seconds -- the CI `fast` job
	$(PY) -m pytest -m "not slow and not network and not container and not artifacts"

test-network: ## Tier 6b live agreement -- nightly only
	$(PY) -m pytest -m network

train: ## Build the canonical artifact bundle
	$(PY) -m phishguard.train --profile corrected

train-legacy: ## Reproduce the notebook's recorded (leaky) metrics
	$(PY) -m phishguard.train --profile legacy

fixtures: ## One-shot live capture of the agreement fixture corpus
	$(PY) scripts/capture_fixtures.py

agreement: ## Re-run the extraction agreement gate against committed fixtures
	$(PY) scripts/agreement_report.py --offline

selftest: ## Verify the installed bundle against the golden row
	$(PY) -m phishguard.selftest --golden

app: ## Run Streamlit locally
	$(PY) -m streamlit run app/Home.py

docker-build: ## Build the deployment image
	docker build -t $(IMAGE) .

docker-test: ## Tier 9: golden row inside the container
	docker run --rm $(IMAGE) python -m phishguard.selftest --golden

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
