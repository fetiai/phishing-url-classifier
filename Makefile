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

test-fast: ## The subset that runs in seconds
	$(PY) -m pytest -m "not slow and not network and not container and not artifacts"

test-network: ## Live agreement against the internet -- run manually
	$(PY) -m pytest -m network

train: ## Build the canonical artifact bundle, honouring the demotion list
	# The agreement report is an input, not a by-product: the demotion list decides what
	# the extractor emits at serving time, so a model fitted without it is fitted on
	# features it will never receive.
	$(PY) -m phishguard.train --profile corrected \
		--agreement artifacts/v1/extraction_agreement.json \
		--demote "$$($(PY) -c "import json,pathlib; p=pathlib.Path('artifacts/v1/extraction_agreement.json'); print(','.join(json.loads(p.read_text()).get('demoted',[])) if p.exists() else '')")"

train-fresh: ## Build a bundle with no demotion list (first run, before the gate)
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

docker-test: ## Verify the built image: golden row, then every page renders
	docker run --rm $(IMAGE) python -m phishguard.selftest --golden
	# Not a health check. Streamlit answers 200 on / and /_stcore/health whether or not
	# the script inside them ran, so a status code cannot tell a working app from one
	# that raises on every request -- which is exactly how a broken import once passed.
	# This runs each page through Streamlit's own script runner under the image's real
	# sys.path, which is the only place a packaging fault like that is visible.
	docker run --rm $(IMAGE) python -c "$$RENDER_CHECK"

define RENDER_CHECK
from streamlit.testing.v1 import AppTest
pages = ['app/Home.py', 'app/pages/1_Single_URL.py', 'app/pages/2_Batch_CSV.py',
         'app/pages/3_Model_Evaluation.py', 'app/pages/4_Dataset_Explorer.py',
         'app/pages/5_Methodology.py']
failed = []
for page in pages:
    at = AppTest.from_file(page, default_timeout=120)
    at.run()
    if at.exception:
        failed.append((page, '; '.join(f'{e.type}: {e.value}' for e in at.exception)))
        print(f'FAIL {page}')
    else:
        print(f'ok   {page}')
if failed:
    for page, detail in failed:
        print(f'  {page}: {detail}')
    raise SystemExit(1)
endef
export RENDER_CHECK

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
