"""Reading and writing the artifact bundle.

The bundle is the interface between training and serving. Everything the application needs
in order to classify is in it, and nothing the application does recomputes any of it.

Two deliberate format choices:

  - The scaler is plain JSON, not a pickled StandardScaler. A pickled estimator arrives
    with a fit_transform method attached, and the entire class of defect this codebase
    exists to correct is someone calling it at serving time. Numbers cannot be re-fitted.
  - The from-scratch Naive Bayes is JSON too, because the feature inspector reads its
    per-class mean and standard deviation tables directly to render contributions. A
    pickle would make the interface depend on unpickling a class it must not import.

Every file is hashed into the manifest and every hash is verified at load. A container
that boots with a bundle it cannot account for is worse than one that refuses to boot: it
serves predictions nobody can trace to a training run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from phishguard import ARTIFACT_SCHEMA_VERSION, __version__, schema
from phishguard.constants import KEYWORD_LIST_VERSION
from phishguard.models.knn_scratch import ScratchKNN
from phishguard.models.nb_scratch import ScratchGaussianNB
from phishguard.models.sk import SklearnGaussianNB, SklearnKNN
from phishguard.preprocess.stats import (
    CategoricalFillStep,
    ClipBound,
    FittedStats,
    HtmlFallback,
    NumericFill,
    ScalerParams,
)

#: JSON has no tuple keys, so a cascade key is joined with a separator that cannot occur
#: in a numeric literal.
KEY_SEPARATOR = "|"


class BundleError(RuntimeError):
    """The bundle is absent, incomplete, or does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _encode_key(key: tuple[float, ...]) -> str:
    return KEY_SEPARATOR.join(repr(float(k)) for k in key)


def _decode_key(encoded: str) -> tuple[float, ...]:
    return tuple(float(part) for part in encoded.split(KEY_SEPARATOR))


# ---------------------------------------------------------------------------
# FittedStats <-> JSON
# ---------------------------------------------------------------------------


def stats_to_json(stats: FittedStats) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "char_prob": stats.char_prob,
        "tld": {
            "prob_mean": stats.tld_prob_mean,
            "global_fill": stats.tld_prob_global_fill,
            "skew": stats.tld_skew,
            "fill_method": stats.tld_fill_method,
        },
        "numeric_fill": [asdict(f) for f in stats.numeric_fill],
        "categorical_cascade": [
            {
                "column": s.column,
                "groupby_cols": list(s.groupby_cols),
                "modes": {_encode_key(k): v for k, v in s.modes.items()},
                "global_mode": s.global_mode,
            }
            for s in stats.categorical_cascade
        ],
        "clip_bounds": [asdict(b) for b in stats.clip_bounds],
        "scaler": {
            "columns": list(stats.scaler.columns),
            "mean_": list(stats.scaler.mean_),
            "scale_": list(stats.scaler.scale_),
        },
        "nb_drop": list(stats.nb_drop),
        "nb_drop_reasons": stats.nb_drop_reasons,
        "html_fallbacks": [asdict(f) for f in stats.html_fallbacks],
        "columns": {
            "feature_order": list(stats.feature_order),
            "numerical_columns": list(stats.numerical_columns),
            "continuous_columns": list(stats.continuous_columns),
            "discrete_columns": list(stats.discrete_columns),
            "categorical_columns_filtered": list(stats.categorical_columns_filtered),
        },
        "n_train_rows": stats.n_train_rows,
        "demoted_features": list(stats.demoted_features),
    }


def stats_from_json(payload: dict[str, Any]) -> FittedStats:
    version = payload.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise BundleError(
            f"artifact schema version mismatch: bundle is {version!r}, "
            f"this build expects {ARTIFACT_SCHEMA_VERSION!r}"
        )

    columns = payload["columns"]
    return FittedStats(
        char_prob={str(k): float(v) for k, v in payload["char_prob"].items()},
        tld_prob_mean={str(k): float(v) for k, v in payload["tld"]["prob_mean"].items()},
        tld_prob_global_fill=float(payload["tld"]["global_fill"]),
        tld_skew=float(payload["tld"]["skew"]),
        tld_fill_method=payload["tld"]["fill_method"],
        numeric_fill=tuple(NumericFill(**f) for f in payload["numeric_fill"]),
        categorical_cascade=tuple(
            CategoricalFillStep(
                column=s["column"],
                groupby_cols=tuple(s["groupby_cols"]),
                modes={_decode_key(k): float(v) for k, v in s["modes"].items()},
                global_mode=float(s["global_mode"]),
            )
            for s in payload["categorical_cascade"]
        ),
        clip_bounds=tuple(ClipBound(**b) for b in payload["clip_bounds"]),
        scaler=ScalerParams(
            columns=tuple(payload["scaler"]["columns"]),
            mean_=tuple(float(v) for v in payload["scaler"]["mean_"]),
            scale_=tuple(float(v) for v in payload["scaler"]["scale_"]),
        ),
        nb_drop=tuple(payload["nb_drop"]),
        nb_drop_reasons=payload["nb_drop_reasons"],
        html_fallbacks=tuple(HtmlFallback(**f) for f in payload["html_fallbacks"]),
        feature_order=tuple(columns["feature_order"]),
        numerical_columns=tuple(columns["numerical_columns"]),
        continuous_columns=tuple(columns["continuous_columns"]),
        discrete_columns=tuple(columns["discrete_columns"]),
        categorical_columns_filtered=tuple(columns["categorical_columns_filtered"]),
        n_train_rows=int(payload.get("n_train_rows", 0)),
        demoted_features=tuple(payload.get("demoted_features", ())),
    )


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------


def nb_scratch_to_json(model: ScratchGaussianNB) -> dict[str, Any]:
    if model.log_prior_ is None or model.theta_ is None or model.sigma_ is None:
        raise BundleError("cannot serialise an unfitted Naive Bayes")
    return {
        "classes": model.classes_.tolist(),
        "log_prior": model.log_prior_.tolist(),
        "theta": model.theta_.tolist(),
        "sigma": model.sigma_.tolist(),
    }


def nb_scratch_from_json(payload: dict[str, Any]) -> ScratchGaussianNB:
    model = ScratchGaussianNB()
    model.classes_ = np.asarray(payload["classes"], dtype=np.int64)
    model.log_prior_ = np.asarray(payload["log_prior"], dtype=np.float64)
    model.theta_ = np.asarray(payload["theta"], dtype=np.float64)
    model.sigma_ = np.asarray(payload["sigma"], dtype=np.float64)
    return model


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


class ArtifactBundle:
    """Everything the application loads at start-up, in one object."""

    def __init__(
        self,
        stats: FittedStats,
        knn_scratch: ScratchKNN,
        knn_sklearn: SklearnKNN,
        nb_scratch: ScratchGaussianNB,
        nb_sklearn: SklearnGaussianNB,
        metrics: dict[str, Any],
        manifest: dict[str, Any],
        feature_reference: dict[str, Any],
        agreement: dict[str, Any],
        golden_row: dict[str, Any],
    ) -> None:
        self.stats = stats
        self.knn_scratch = knn_scratch
        self.knn_sklearn = knn_sklearn
        self.nb_scratch = nb_scratch
        self.nb_sklearn = nb_sklearn
        self.metrics = metrics
        self.manifest = manifest
        self.feature_reference = feature_reference
        self.agreement = agreement
        self.golden_row = golden_row

    @property
    def models(self) -> dict[str, Any]:
        return {
            "knn_scratch": self.knn_scratch,
            "knn_sklearn": self.knn_sklearn,
            "nb_scratch": self.nb_scratch,
            "nb_sklearn": self.nb_sklearn,
        }

    @property
    def demoted(self) -> frozenset[str]:
        return frozenset(self.stats.demoted_features)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys and a fixed separator make the bytes a deterministic function of the
    # content, so refitting with the same data reproduces the same hash.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_manifest(directory: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(directory))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "phishguard_version": __version__,
        "keyword_list_version": KEYWORD_LIST_VERSION,
        "git_sha": _git_sha(),
        "feature_count": len(schema.FEATURE_ORDER),
        "files": files,
    }
    manifest.update(extra or {})
    return manifest


def verify_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Fail loudly when the bundle on disk is not the bundle that was trained."""
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise BundleError(
            f"artifact schema version mismatch: bundle {manifest.get('artifact_schema_version')!r},"
            f" build {ARTIFACT_SCHEMA_VERSION!r}"
        )
    if manifest.get("keyword_list_version") != KEYWORD_LIST_VERSION:
        raise BundleError(
            "keyword list version mismatch: the frozen keyword lists changed since this "
            "bundle was trained, so its page features mean something different from what "
            "this build would compute"
        )
    if manifest.get("feature_count") != len(schema.FEATURE_ORDER):
        raise BundleError("feature count mismatch between bundle and build")

    for relative, record in manifest.get("files", {}).items():
        path = directory / relative
        if not path.exists():
            raise BundleError(f"bundle file missing: {relative}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise BundleError(
                f"bundle file {relative} does not match its recorded hash "
                f"(expected {record['sha256'][:12]}, found {actual[:12]})"
            )


def load_bundle(directory: Path, *, verify: bool = True) -> ArtifactBundle:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(
            f"no artifact bundle at {directory}. Build one with "
            f"`python -m phishguard.train --profile corrected`."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        verify_manifest(directory, manifest)

    stats = stats_from_json(json.loads((directory / "fitted_stats.json").read_text("utf-8")))

    reference = np.load(directory / "models" / "knn_scratch.npz")
    knn_scratch = ScratchKNN(k=int(reference["k"])).fit(reference["X"], reference["y"])

    knn_sklearn = joblib.load(directory / "models" / "knn_sklearn.joblib")
    nb_sklearn = joblib.load(directory / "models" / "nb_sklearn.joblib")
    nb_scratch = nb_scratch_from_json(
        json.loads((directory / "models" / "nb_scratch.json").read_text("utf-8"))
    )

    def read(name: str) -> dict[str, Any]:
        path = directory / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    return ArtifactBundle(
        stats=stats,
        knn_scratch=knn_scratch,
        knn_sklearn=knn_sklearn,
        nb_scratch=nb_scratch,
        nb_sklearn=nb_sklearn,
        metrics=read("metrics.json"),
        manifest=manifest,
        feature_reference=read("feature_reference_stats.json"),
        agreement=read("extraction_agreement.json"),
        golden_row=read("golden_row.json"),
    )
