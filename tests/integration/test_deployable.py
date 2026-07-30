"""The bundle a deploy host will actually receive.

Deploy hosts build from the *committed* repository. The image bakes the artifact bundle in
rather than mounting it, so anything not tracked by git is absent from the build context and
`COPY artifacts/` fails -- which is exactly how a deploy broke while the local build, the
whole test suite, and a healthy container all stayed green. The local build context is the
working directory, so it had the bundle on disk; the host only ever had what was committed.

The lesson generalises: a check that reads the filesystem cannot tell you what a build from
git will contain. These tests ask git.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from phishguard import schema
from phishguard.artifacts import load_bundle
from phishguard.verify_bundle import REQUIRED_TRACKED, verify

ARTIFACTS = Path("artifacts/v1")
AGREEMENT = Path("extraction_agreement.json")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, timeout=30
    ).stdout


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    return {line.strip() for line in _git("ls-files").splitlines() if line.strip()}


@pytest.mark.parametrize("relative", REQUIRED_TRACKED)
def test_bundle_file_is_committed(tracked, relative):
    """The check that would have caught the failed deploy."""
    assert relative in tracked, (
        f"{relative} is not tracked. The deploy host builds from git, so it will not exist "
        f"in the build context and COPY artifacts/ will fail."
    )


def test_the_dockerfile_copies_only_what_is_committed(tracked):
    """Every COPY source in the Dockerfile must be present in a fresh clone."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    sources = []
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY ") or "--from=" in stripped:
            continue
        parts = stripped.split()[1:-1]  # drop COPY and the destination
        sources.extend(parts)

    for source in sources:
        path = Path(source.rstrip("/"))
        assert path.exists(), f"Dockerfile copies {source}, which is not in the repository"
        if path.is_dir():
            assert any(t.startswith(f"{path}/") for t in tracked), (
                f"Dockerfile copies {source}/ but git tracks nothing under it, so a build "
                f"from a clean clone will fail"
            )
        else:
            assert str(path) in tracked, f"Dockerfile copies untracked {source}"


def test_agreement_report_is_committed(tracked):
    """It is a training *input*. Untracked means a from-source build demotes nothing."""
    assert str(AGREEMENT) in tracked


def test_committed_bundle_matches_the_committed_agreement_report():
    """Guards the drift that matters: a bundle trained against a different demotion list
    than the one under review."""
    report = json.loads(AGREEMENT.read_text(encoding="utf-8"))
    bundle = load_bundle(ARTIFACTS)
    assert set(bundle.stats.demoted_features) == set(report.get("demoted", ()))


def test_demoted_features_are_all_real_page_features():
    report = json.loads(AGREEMENT.read_text(encoding="utf-8"))
    unknown = set(report.get("demoted", ())) - set(schema.HTML_FEATURES)
    assert not unknown, f"agreement report demotes non-page features: {sorted(unknown)}"


def test_verify_bundle_passes():
    """The same gate `make verify-bundle` runs before a deploy."""
    assert verify() == []


def test_no_credentials_are_tracked(tracked):
    """A `git add -A` once swept a private deploy key into a commit. Nothing about that was
    visible in a diff summary, and only rewriting history got it back out."""
    suspicious = [
        path
        for path in tracked
        if path.startswith(".ssh/")
        or path.endswith((".pem", "_rsa", "_ed25519"))
        or Path(path).name.startswith("id_")
    ]
    assert not suspicious, f"credential-looking files are tracked: {sorted(suspicious)}"


def test_no_tracked_file_contains_a_private_key(tracked):
    """Belt and braces: catches a key committed under a name the pattern above misses."""
    markers = ("BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "BEGIN PRIVATE KEY")
    offenders = []
    for path in tracked:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size > 64_000:
            continue
        try:
            head = candidate.read_text(encoding="utf-8", errors="ignore")[:400]
        except OSError:
            continue
        if any(marker in head for marker in markers):
            offenders.append(path)
    assert not offenders, f"tracked files contain private key material: {sorted(offenders)}"
