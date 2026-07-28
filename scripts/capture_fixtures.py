"""Capture the agreement fixture corpus. Run once; the output is committed.

The dataset references a crawled HTML corpus by filename, and that corpus is not in this
repository -- so the original pages cannot be replayed. The nearest available substitute
is to re-fetch a stratified sample of the URLs now, store what comes back, and compare
against the values the dataset recorded for those same rows.

Fetching goes through the production guarded client, not a bare requests call. If the
guard would refuse a URL in production it must refuse it here too, otherwise the corpus is
built from pages the running service could never have seen.

The result is committed and becomes the regression baseline: from then on the agreement
harness runs in CI against these bytes with no network at all, so the measurement stops
depending on the internet's mood.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import time
from pathlib import Path

import pandas as pd

from phishguard import schema
from phishguard.config import FetchConfig
from phishguard.fetch import client

FIXTURE_DIR = Path("tests/fixtures/html")
EXPECTED = Path("tests/fixtures/expected.parquet")


def sample_urls(source: Path, n: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(source, low_memory=False)
    frame = frame[frame["URL"].notna()]

    # Stratified, because the two classes fail in completely different ways: legitimate
    # URLs mostly still resolve, phishing URLs mostly do not. A sample dominated by either
    # would misrepresent what the extractor faces.
    per_class = n // 2
    parts = [
        frame[frame["label"] == label].sample(
            n=min(per_class, int((frame["label"] == label).sum())), random_state=seed
        )
        for label in (schema.PHISHING_LABEL, schema.LEGITIMATE_LABEL)
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds between requests; politeness, not performance",
    )
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    config = FetchConfig(enabled=True, robots=False)

    candidates = sample_urls(args.source, args.n, args.seed)
    print(f"attempting {len(candidates)} URLs")

    captured = []
    outcomes: dict[str, int] = {}

    for i, row in enumerate(candidates.itertuples(), start=1):
        result = client.get(row.URL, config)
        outcomes[result.outcome.value] = outcomes.get(result.outcome.value, 0) + 1
        print(f"  [{i}/{len(candidates)}] {result.outcome.value:20} {row.URL[:70]}")

        if result.ok and result.html:
            sha1 = hashlib.sha1(result.html.encode("utf-8")).hexdigest()
            (FIXTURE_DIR / f"{sha1}.html.gz").write_bytes(
                gzip.compress(result.html.encode("utf-8"))
            )

            record = {
                "sha1": sha1,
                "url": row.URL,
                "final_url": result.final_url,
                "label": int(row.label),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # The dataset's own values for this row -- the thing being compared against.
            for name in schema.HTML_FEATURES:
                record[name] = getattr(row, name, None)
            captured.append(record)

        time.sleep(args.delay)

    if not captured:
        raise SystemExit(
            "nothing was captured. Every URL failed, which is itself the finding: at this "
            "age most of the corpus is unreachable, and the agreement gate cannot run."
        )

    frame = pd.DataFrame(captured)
    frame.to_parquet(EXPECTED, index=False)

    print(f"\ncaptured {len(frame)} pages into {FIXTURE_DIR}")
    print(f"  legitimate: {int((frame['label'] == schema.LEGITIMATE_LABEL).sum())}")
    print(f"  phishing:   {int((frame['label'] == schema.PHISHING_LABEL).sum())}")
    print(f"\noutcomes: {outcomes}")
    print(
        "\nThe phishing capture rate is expected to be low. Those domains are "
        "short-lived and this dataset was crawled in 2023-24, so most are gone. That is "
        "why the gate is judged on legitimate URLs."
    )


if __name__ == "__main__":
    main()
