# Contributing

## Definition of done

A change is not done until all of these hold. They are listed in the order they tend to
fail.

- [ ] `make lint type test` is green.
- [ ] New behaviour has a test that fails without the change. A test written after the fact
      that passes either way is documentation, not evidence.
- [ ] No `TODO`, no `pytest.mark.skip`, no `.only`, no stub returning a placeholder. An
      unimplemented branch is a blocker to report, not a step to hide.
- [ ] Numbers in prose match numbers in code. If you changed a metric, change every place
      that quotes it — the README, the app copy, and the tests.

## Rules that are not negotiable

**`transform()` computes no statistic.** No `.mean()`, `.median()`, `.mode()`, `.std()`,
`.skew()`, `.quantile()`, `.value_counts()`, `.groupby().agg()`, `fit()` or
`fit_transform()` on the input frame. Anything depending on more than the current row is
learned in `fit()` and persisted.

The row/batch invariance test proves it, and it compares **bitwise**. Do not relax it to
`allclose` to make a change pass — a tolerance would accept exactly the defect the test
exists to catch, because a leaked statistic computed from a batch that resembles the
training set produces *nearly* the right answer.

**The ported feature functions are frozen.** The 26 `fill_*` functions in
`phishguard/features/url_features.py` preserve the original's behaviour exactly, including
behaviour that is clearly wrong. They define what the training data means; changing one
shifts the distribution the models were fitted on and no metric will tell you. If one is
genuinely broken, that is a retraining decision with a metric-invalidation cost, not a
bugfix.

`tests/legacy_reference.py` and `notebooks/` are never edited. They are the oracle. Editing
either makes the fidelity test compare the port against itself.

**Extractors return `None`, never a sentinel.** A zero count is a measurement; an absent
one is not. Once `0` or `-1` reaches the imputer the two are indistinguishable, and "we
could not look" silently becomes evidence.

**Only `phishguard/fetch/` opens a socket.** `features/` is pure. `phishguard/` never
imports `streamlit`; `app/` never imports `sklearn`.

**Changing a frozen list invalidates the models.** The keyword and domain lists in
`constants.py` are versioned and checked at bundle load. Change one and you must retrain
and bump `KEYWORD_LIST_VERSION`.

## Touching the fetch guard

Changes to `phishguard/fetch/safety.py` or `client.py` need the SSRF table extended, not
just kept passing. When you add a capability, add the case that proves it cannot be abused.

Two invariants that are easy to break without noticing:

- **Judge the resolved address, never the hostname.** `localtest.me` is an ordinary public
  DNS record pointing at `127.0.0.1`.
- **Connect to the address you validated.** Validating a name and then letting the HTTP
  stack resolve it again leaves the rebinding window open and makes the check decorative.

Raising `FETCH_MAX_BYTES` or `FETCH_CONCURRENCY` is a memory decision: in-flight usage is
their product.

## Changing what the app claims

The honesty affordances — the coverage meter, the provenance table, the abstention rule,
the parity delta, the baseline shown beside every accuracy — are load-bearing, not
decoration. Removing one makes the application overstate what it knows.

In particular: never present the four models as four opinions. They are two algorithms
implemented twice, and averaging a model with its own reference is double-counting.
