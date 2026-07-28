"""Build the committed raw sample used by the fidelity and invariance tests.

Deliberately over-samples rows with a missing URL (31% of the corpus) and a missing Domain
(50%), because those are the branches the ported functions are most likely to get wrong
and the least likely to be exercised by a naive random draw. A sample that only contains
well-formed rows would let a broken NaN branch pass every test.

Run once; the output is committed. Regenerating it changes the oracle the fidelity tests
compare against, so it is not part of any routine workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEED = 42
DEFAULT_N = 2000


def build_sample(src: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(src, low_memory=False)

    url_missing = df["URL"].isna()
    domain_missing = df["Domain"].isna()

    strata = {
        "url_missing": df[url_missing],
        "domain_missing_url_present": df[domain_missing & ~url_missing],
        "both_present_phishing": df[~url_missing & ~domain_missing & (df["label"] == 0)],
        "both_present_legitimate": df[~url_missing & ~domain_missing & (df["label"] == 1)],
    }

    # A quarter from each stratum, so the two NaN branches are ~50% of the sample rather
    # than the ~31%/50% they would be by chance in a single column.
    per_stratum = n // len(strata)
    parts = []
    for name, frame in strata.items():
        if frame.empty:
            raise RuntimeError(f"stratum {name!r} is empty; the source data is not what we expect")
        take = min(per_stratum, len(frame))
        parts.append(frame.sample(n=take, random_state=seed))

    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/raw_sample_2000.parquet"))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    sample = build_sample(args.src, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)

    print(f"wrote {len(sample)} rows to {args.out}")
    print(f"  URL missing:    {sample['URL'].isna().sum()}")
    print(f"  Domain missing: {sample['Domain'].isna().sum()}")
    print(f"  label==0:       {(sample['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
