# The original notebook — frozen

`18222023_18222063_18222056_18222059_Tubes2.ipynb` is the submitted coursework notebook. It
is kept here as **historical provenance and nothing else**.

## It is never edited

A CI job asserts this file is unchanged on every pull request.

The reason is not sentiment. A verbatim copy of the notebook's feature functions lives in
`tests/legacy_reference.py` and is the oracle the production port is asserted bit-identical
to. Editing the notebook would invite someone to "fix" that copy to match, and the test
would then be comparing the port against itself — passing while proving nothing.

## `phishguard/train.py` is the source of truth

Everything the deployed service uses is produced by `python -m phishguard.train`. The
notebook produces nothing the service loads, and it never could: `import pickle` appears in
it once (cell 243) and is never called, so every model it fitted died with its kernel.

## What it got wrong

Twelve defects were found reading it, and they are documented in full on the application's
Methodology page. The two that matter most:

- `DataPreparation.fit()` was `return self`. Every statistic — the character frequency
  table, per-TLD means, imputation values, clip bounds, the scaler — was recomputed inside
  `transform()` from whatever batch it was handed. A single-row input would derive its
  statistics from that single row.
- `apply_standardization` built a fresh `StandardScaler` internally, so the training and
  validation splits were each standardised by their own mean and standard deviation. The
  scaling behind every recorded metric therefore does not exist at serving time.

Correcting the second one costs about nine points of phishing recall. That is not a
criticism of the notebook as coursework; it is the reason its numbers cannot be published
as this system's results, and why both the recorded and the corrected figures are reported
side by side.

## Reproducing the recorded numbers

```bash
python -m phishguard.train --profile legacy
```

This reconstructs the original configuration on purpose — k=6, a 10,000-row reference set,
and the separate-fit scaling — and lands within 0.0002 of the recorded accuracy. That
agreement is what shows the port preserved the arithmetic; without it, the corrected
numbers could not be attributed to the fix rather than to the rewrite.
