# PhishGuard

Phishing URL classification with k-Nearest Neighbours and Gaussian Naive Bayes — each
implemented twice, from scratch and with scikit-learn — served as a single Streamlit
application.

> **This is a coursework reimplementation, not a security product.** It is trained on a
> static 2023–24 dataset, has no threat intelligence, no blocklist, and no knowledge of any
> campaign newer than its training data. Do not use it to decide whether a link is safe.

IF3070 Foundations of Artificial Intelligence, Tugas Besar 2 — Kelompok 16:
Thalita Zahra Sutejo (18222023), Irfan Musthofa (18222056), Eleanor Cordelia (18222059),
Muhammad Faiz Atharrahman (18222063).

---

## What it does

Paste a URL. The service fetches the page under an SSRF guard, extracts 49 features, and
scores it with four models. **21 features come from the URL string; 28 need the page.**
When too little of the page can be read, it says so rather than guessing — see
[Why it sometimes refuses to answer](#why-it-sometimes-refuses-to-answer).

| Page | Purpose |
|---|---|
| Single URL | Fetch, classify, and inspect every feature with its provenance |
| Batch | Score a feature CSV, or a capped list of URLs |
| Model evaluation | Confusion matrices, threshold sweep, as-corrected beside as-recorded |
| Dataset explorer | Class balance, missingness, per-class distributions |
| Methodology | The defect ledger, the port record, and the extraction agreement report |

## Quickstart

```bash
make install          # venv + pinned dependencies
make train            # build the artifact bundle (~3 minutes)
make app              # http://localhost:8501
```

`make test` runs the full offline suite; `make lint type` runs ruff and mypy.

### With Docker

```bash
export SITE_ADDRESS=phishguard.example.com
export FETCH_SELF_IPS=203.0.113.10        # this host's public address — required
docker compose up -d --build
```

The image **bakes the artifact bundle in** rather than mounting it, so the image tag is a
complete description of what the service will predict, and rolling back is deploying the
previous tag. The build fails if the bundle does not reproduce its own golden row.

## Results

Measured on a 28,081-row validation split. **The held-out file shipped with the dataset has
no labels, so there is no test score and none is claimed.**

| Model | Phishing recall | Phishing precision | Accuracy |
|---|---|---|---|
| KNN, from scratch | 0.759 | 0.981 | 0.98045 |
| KNN, scikit-learn | 0.763 | 0.981 | 0.98073 |
| Gaussian NB, from scratch | 0.788 | 0.911 | 0.97789 |
| Gaussian NB, scikit-learn | 0.888 | 0.877 | 0.98191 |

**Read every accuracy against 0.9248.** The corpus is 92.48% legitimate, so answering
"legitimate" to everything scores 0.9248 while catching no phishing whatsoever. Accuracy
alone cannot tell a working detector from a constant; phishing recall can, which is why it
leads every report in this project.

Class 0 is phishing and is the positive class throughout.

### As-recorded versus as-corrected

The original notebook reported **0.98643 accuracy and 0.857 phishing recall** for its
scikit-learn KNN. That number came from a pipeline that standardised the validation split
by *its own* mean and standard deviation — information no deployed model can have, since
there is no batch to average over when someone submits a single URL.

`--profile legacy` reconstructs that configuration deliberately and reproduces it to within
0.0002 (0.98629 accuracy, 0.854 recall), which is what demonstrates the port did not change
the arithmetic. Correcting the leak costs about **nine points of phishing recall**
(0.854 → 0.763). That gap is the honest measure of how much of the original result was
leakage rather than detection.

Both profiles ship in `metrics.json`. The leaky one is flagged `"leaky": true` and is never
presented as this system's result.

## Why it sometimes refuses to answer

Below 60% page-feature coverage, no verdict is given.

Imputation fills missing features from the training distribution, which is 92.5%
legitimate. A failed fetch therefore does not yield a *neutral* prediction — it yields one
biased toward **legitimate**, the exact wrong direction for a phishing detector.

This is not an edge case. Phishing domains are short-lived, so a phishing URL from a
2023–24 crawl is usually dead, parked, or on registrar hold today. "Could not reach it"
correlates with phishing in reality and with legitimate in the imputed features. Saying
*not enough evidence* is the only defensible answer.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FETCH_ENABLED` | `true` | Kill switch. `false` degrades to URL-only features without a rebuild |
| `FETCH_SELF_IPS` | *(required)* | This host's public addresses. Empty leaves the fetcher able to reach the app itself |
| `FETCH_DENY_HOSTS` | *(empty)* | Operational blocklist; covers subdomains |
| `FETCH_ROBOTS` | `true` | Whether the `Robots` feature costs a second request to `/robots.txt` |
| `FETCH_TIMEOUT_S` | `10` | Wall-clock budget across all redirect hops |
| `FETCH_MAX_BYTES` | `2097152` | Cap on **decompressed** body bytes |
| `FETCH_MAX_REDIRECTS` | `3` | Hops followed, each revalidated in full |
| `FETCH_CONCURRENCY` | `8` | Process-wide outbound limit |
| `FETCH_RATE_PER_SESSION` | `10` | Fetches per session per window |
| `FETCH_RATE_WINDOW_S` | `300` | Rate-limit window |
| `FETCH_BATCH_MAX_URLS` | `25` | Cap on URL-list batches |
| `REF_SCOPE` | `anchor` | Which definition the reference-count features use |
| `COVERAGE_MIN_RATIO` | `0.60` | Coverage below which the app abstains |
| `ARTIFACTS_DIR` | `artifacts/v1` | Where the bundle is loaded from |

In-flight memory is bounded by `FETCH_CONCURRENCY × FETCH_MAX_BYTES` — 16 MiB at these
defaults. Raising both without raising the container's memory limit is how it gets
OOM-killed.

## Layout

```
phishguard/          the library — never imports streamlit
  schema.py          the frozen 49 features and every column group
  features/          url_features.py (ported), html_features.py (new), extract.py
  preprocess/        stats.py, scaler.py, transformer.py — owns all fitted state
  models/            knn_scratch.py, nb_scratch.py, sk.py
  fetch/             safety.py, client.py — the only module that opens a socket
  train.py           the single source of truth for how models are produced
app/                 Streamlit — never imports sklearn
notebooks/           the original notebook, frozen
```

Four rules hold it together: the library never imports Streamlit; `features/` is pure and
emits `NaN` for anything it cannot determine; `fetch/` is the only module that opens a
socket; and `preprocess/` owns every fitted statistic and decides what a `NaN` becomes.

## What was wrong with the original, and what changed

The notebook this replaces saved no model of any kind — `import pickle` appears once and is
never called — so everything it computed died with its kernel. Reading it closely surfaced
twelve defects. Two matter most:

**Every statistic was recomputed at transform time.** `fit()` was `return self`, so the
character frequency table, per-TLD means, imputation values, clip bounds and scaler were
all derived from whatever batch arrived. One row in meant statistics computed from that one
row.

**Each split was scaled by its own statistics**, so the scaling behind every reported number
does not exist at serving time and cannot be reconstructed.

The fix is a real fit/transform split, and the evidence is a test: transforming rows
individually must equal transforming them as a batch, **bitwise on float32**. A
tolerance-based comparison would pass even while a statistic leaked from a favourable
batch; exact equality can only hold if `transform` reads its numbers from persisted state.
The full ledger is on the Methodology page and in `tests/unit/test_invariance.py`.

The 26 URL feature functions are ported **verbatim, quirks included**. One obfuscation rule
reverses the URL and tests a pattern that is unchanged by reversal, so it fires on any bare
domain — almost certainly not the intent. It is preserved anyway: those functions define
what the training data means, and changing one would shift the distribution the models were
fitted on with no metric revealing it. A frozen copy of the originals lives in
`tests/legacy_reference.py`, and the port is asserted bit-identical to it over 2,000 rows.

## The largest open question

**The 25 page-feature extraction rules are hypotheses.** The dataset's own extraction code
is not published, and neither is the HTML corpus its rows were computed from. The rules
here are reconstructions inferred from feature names, and reading them cannot establish
whether they match.

So they are measured rather than trusted:

```bash
python scripts/capture_fixtures.py     # once; the corpus is committed
python scripts/agreement_report.py     # offline, no network
```

Each feature is gated — binary features at 0.85 agreement, count features at 0.70
Spearman ρ — and anything that misses its gate is **demoted**: its extractor returns `None`
permanently, its value is always imputed, and the interface labels it as not reliably
extractable. Results land in `extraction_agreement.json` and render on the Methodology page.

Agreement is judged on legitimate URLs only. On phishing URLs the comparison mostly measures
link rot: those domains are largely dead, so what gets fetched today is a parking page
rather than what the dataset saw.

## Testing

```bash
make test         # everything offline
make test-fast    # the subset that runs in seconds
```

The tests that carry the weight:

- **Port fidelity** — every ported function against the frozen original, `check_exact=True`.
- **Row/batch invariance** — the core correctness property, bitwise.
- **Models** — vectorised KNN against the naive loop *including engineered exact ties*;
  log-space Naive Bayes against the raw product, showing where the product collapses.
- **Fetch safety** — the SSRF rejection table: a public URL redirecting to loopback, a
  hostname resolving to `127.0.0.1`, encoded loopback literals, and a compression bomb.
- **Bundle and inference** — hash verification, tamper detection, the golden row, and four
  verdicts with a fully populated provenance table.

## Licence and data

The dataset is the UCI PhiUSIIL Phishing URL Dataset (ID 967). The report in `doc/` and the
notebook in `notebooks/` are coursework artifacts; the notebook is frozen provenance and is
never edited.
