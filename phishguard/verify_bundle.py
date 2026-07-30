"""Check that the committed artifact bundle is deployable.

This exists because of a specific failure. The deployment target builds from the committed
repository, the image bakes the bundle in, and the bundle was gitignored -- so the build
context on the deploy host had no ``artifacts/`` at all and ``COPY`` failed. Locally the
build context is the working directory, which had the bundle sitting on disk, so the build
succeeded here and could never have succeeded there.

Nothing about that was visible from a local build, a passing test suite, or a healthy
container. What makes it visible is asking git, not the filesystem, what the deploy host
will actually receive.

Run via ``make verify-bundle`` before deploying.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from phishguard.artifacts import BundleError, load_bundle
from phishguard.config import APP

AGREEMENT = Path("extraction_agreement.json")

#: Files the image needs. Absent from git means absent from the deploy build context.
REQUIRED_TRACKED = (
    "artifacts/v1/manifest.json",
    "artifacts/v1/fitted_stats.json",
    "artifacts/v1/metrics.json",
    "artifacts/v1/golden_row.json",
    "artifacts/v1/models/knn_scratch.npz",
    "artifacts/v1/models/knn_sklearn.joblib",
    "artifacts/v1/models/nb_scratch.json",
    "artifacts/v1/models/nb_sklearn.joblib",
)


def _tracked_files() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "artifacts", "extraction_agreement.json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise BundleError(f"could not ask git what is tracked: {exc}") from exc
    return {line.strip() for line in out.splitlines() if line.strip()}


def verify() -> list[str]:
    failures: list[str] = []

    # 1. Is it committed? This is the check that would have caught the deploy failure.
    tracked = _tracked_files()
    for relative in REQUIRED_TRACKED:
        if relative not in tracked:
            failures.append(
                f"{relative} is not tracked by git. The deployment host builds from the "
                f"committed repository, so it will not be in the build context and "
                f"`COPY artifacts/` will fail."
            )

    if not AGREEMENT.exists():
        failures.append(
            f"{AGREEMENT} is missing. Without it, training silently demotes nothing and "
            f"the models are fitted on page features the extractor will never emit."
        )
    elif str(AGREEMENT) not in tracked:
        failures.append(f"{AGREEMENT} exists but is not tracked by git.")

    # 2. Is it intact, and does it still reproduce its own golden row?
    try:
        bundle = load_bundle(APP.artifacts_dir, verify=True)
    except (BundleError, FileNotFoundError) as exc:
        failures.append(f"bundle does not load: {exc}")
        return failures

    # 3. Does the bundle agree with the demotion list it was supposedly trained against?
    if AGREEMENT.exists():
        report = json.loads(AGREEMENT.read_text(encoding="utf-8"))
        expected = set(report.get("demoted", ()))
        actual = set(bundle.stats.demoted_features)
        if expected != actual:
            failures.append(
                "the committed bundle was trained with a different demotion list than the "
                f"committed agreement report. Report demotes {sorted(expected)}; bundle "
                f"records {sorted(actual)}. Re-run `make train` and commit the result."
            )

    return failures


def main() -> int:
    try:
        failures = verify()
    except BundleError as exc:
        print(f"VERIFY FAILED: {exc}", file=sys.stderr)
        return 1

    if failures:
        print("VERIFY FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    bundle = load_bundle(APP.artifacts_dir, verify=True)
    print(
        f"bundle OK: tracked, intact, reproduces its golden row, "
        f"{len(bundle.demoted)} page feature(s) demoted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
