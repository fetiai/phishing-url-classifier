"""Build the artifact bundle. One command, every artifact.

This module is the single source of truth for how the served models are produced. The
notebook is historical provenance and nothing else -- it cannot be, since everything it
computed died with its kernel and it saved no model of any kind.

Data is read from the committed CSVs. The original loaded from a Google Drive link and a
personal file host, which means its inputs were mutable, unversioned, and unavailable to
anyone re-running it later.

TWO PROFILES
============

``corrected`` is canonical. It uses the real fit/transform split, so the statistics
applied at serving time are the statistics learned at training time.

``legacy`` deliberately reconstructs the original's leaky configuration -- k=6 for the
scikit-learn KNN, a 10,000-row reference set, and the validation split standardised by its
own mean and standard deviation. It exists solely to demonstrate that the port can still
reproduce the historical numbers, so that the corrected numbers can be attributed to the
fix rather than to the rewrite. Its metrics are recorded with ``"leaky": true`` and are
never presented as this system's results.

Removing leakage usually lowers scores. The corrected figure is not predicted in advance
and is not compared unfavourably to a number that was measuring something else.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phishguard import ARTIFACT_SCHEMA_VERSION, schema
from phishguard.artifacts import (
    build_manifest,
    nb_scratch_to_json,
    stats_to_json,
    write_json,
)
from phishguard.evaluate import (
    average_precision_phishing,
    compute_metrics,
    disagreement_rate,
)
from phishguard.models.knn_scratch import ScratchKNN
from phishguard.models.nb_scratch import ScratchGaussianNB
from phishguard.models.sk import SklearnGaussianNB, SklearnKNN
from phishguard.preprocess.scaler import SubsetStandardScaler
from phishguard.preprocess.transformer import Preprocessor

log = logging.getLogger("phishguard.train")

TRAIN_CSV = Path("data/raw/train.csv")
DEFAULT_OUT = Path("artifacts/v1")

RANDOM_STATE = 42
TEST_SIZE = 0.2

#: Matches the original's reference size. Kept because the KNN reference set is carried in
#: memory and copied into the image, and 10,000 x 49 float32 is 1.96 MB against 22 MB for
#: the full training split -- a real difference on a small VPS.
KNN_REFERENCE_ROWS = 10_000
KNN_K_CORRECTED = 20
KNN_K_LEGACY = 6


def load_raw(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"training data not found at {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = set(schema.RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"training CSV is missing columns: {sorted(missing)}")

    y = frame[schema.TARGET_COLUMN]

    # The feature columns are deliberately holed with missing values and the pipeline
    # imputes them. The label is not, and a missing one is unrecoverable: casting NaN to
    # int64 yields a garbage sentinel that would be trained on as though it were a class.
    missing_labels = int(y.isna().sum())
    if missing_labels:
        raise ValueError(
            f"{missing_labels} row(s) in {path} have no label. Training on them would "
            f"silently invent a class. Fix or drop the rows before training."
        )

    unexpected = set(y.dropna().unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"label column contains values outside {{0, 1}}: {sorted(unexpected)}")

    X = frame.drop(columns=[*schema.DROPPED_IDENTIFIER_COLUMNS, schema.TARGET_COLUMN])
    return X, y


def split(X: pd.DataFrame, y: pd.Series) -> tuple[Any, Any, Any, Any]:
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _feature_reference_stats(matrix: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Per-feature distribution summary, so the interface never touches the training CSV.

    Powers the "is this value unusual?" column and the per-class z-score in the feature
    inspector.
    """
    out: dict[str, Any] = {}
    phishing = y == schema.PHISHING_LABEL
    for i, name in enumerate(schema.FEATURE_ORDER):
        column = matrix[:, i].astype(np.float64)
        out[name] = {
            "min": float(column.min()),
            "p1": float(np.percentile(column, 1)),
            "p25": float(np.percentile(column, 25)),
            "median": float(np.percentile(column, 50)),
            "p75": float(np.percentile(column, 75)),
            "p99": float(np.percentile(column, 99)),
            "max": float(column.max()),
            "mean_phishing": float(column[phishing].mean()),
            "std_phishing": float(column[phishing].std()),
            "mean_legitimate": float(column[~phishing].mean()),
            "std_legitimate": float(column[~phishing].std()),
        }
    return out


def _dataset_profile(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    return {
        "n_rows": int(len(X)),
        "n_features": len(schema.FEATURE_ORDER),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "class_balance": {str(k): float(v) for k, v in y.value_counts(normalize=True).items()},
        "nan_rate": {
            column: float(X[column].isna().mean()) for column in X.columns
        },
    }


def evaluate_all(
    models: dict[str, Any], X_val: np.ndarray, y_val: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    results: dict[str, Any] = {}
    predictions = {"y_true": y_val}

    for key, model in models.items():
        started = time.monotonic()
        y_pred = model.predict(X_val)
        elapsed = time.monotonic() - started
        scores = model.score_phishing(X_val)

        metrics = compute_metrics(y_val, y_pred)
        results[key] = {
            "name": model.name,
            "family": model.family,
            "is_scratch": model.is_scratch,
            "predict_seconds": elapsed,
            "average_precision_phishing": average_precision_phishing(y_val, scores),
            **metrics.to_dict(),
        }
        predictions[f"{key}_pred"] = y_pred
        predictions[f"{key}_score"] = scores

    # The parity delta the interface shows. A from-scratch model is only evidence of
    # understanding if somebody measured how far it lands from its reference.
    results["_parity"] = {
        "knn": disagreement_rate(predictions["knn_scratch_pred"], predictions["knn_sklearn_pred"]),
        "naive_bayes": disagreement_rate(
            predictions["nb_scratch_pred"], predictions["nb_sklearn_pred"]
        ),
    }

    return results, pd.DataFrame(predictions)


def train(
    profile: str,
    source: Path,
    out_dir: Path,
    demoted: frozenset[str] = frozenset(),
    with_legacy: bool = True,
    agreement: Path | None = None,
) -> dict[str, Any]:
    log.info("loading %s", source)
    X, y = load_raw(source)
    X_train, X_val, y_train, y_val = split(X, y)
    log.info("split: %d train / %d validation", len(X_train), len(X_val))

    preprocessor = Preprocessor()
    log.info("fitting preprocessor")
    stats = preprocessor.fit(X_train.copy(), y_train.copy())

    log.info("transforming training split")
    train_matrix = preprocessor.transform_matrix(X_train.copy())
    log.info("transforming validation split")
    val_matrix = preprocessor.transform_matrix(X_val.copy())

    y_train_array = y_train.to_numpy(dtype=np.int64)
    y_val_array = y_val.to_numpy(dtype=np.int64)

    if profile == "legacy":
        # Reproduce the original's separate-fit standardisation: each split z-scored by
        # its own mean and standard deviation. This is the leak, reconstructed on purpose.
        legacy_scaler = SubsetStandardScaler(schema.NUMERICAL_COLUMNS)
        train_frame = preprocessor.transform(X_train.copy())
        val_frame = preprocessor.transform(X_val.copy())
        train_matrix = SubsetStandardScaler.apply(
            train_frame, legacy_scaler.fit(train_frame)
        ).to_numpy(dtype=np.float32)
        val_matrix = SubsetStandardScaler.apply(
            val_frame, legacy_scaler.fit(val_frame)
        ).to_numpy(dtype=np.float32)
        knn_k = KNN_K_LEGACY
    else:
        knn_k = KNN_K_CORRECTED

    reference_rows = min(KNN_REFERENCE_ROWS, len(train_matrix))
    knn_reference = train_matrix[:reference_rows]
    knn_reference_labels = y_train_array[:reference_rows]

    log.info("fitting models (knn k=%d, reference rows=%d)", knn_k, reference_rows)
    models: dict[str, Any] = {
        "knn_scratch": ScratchKNN(k=knn_k).fit(knn_reference, knn_reference_labels),
        "knn_sklearn": SklearnKNN(k=knn_k).fit(knn_reference, knn_reference_labels),
        "nb_scratch": ScratchGaussianNB().fit(train_matrix, y_train_array),
        "nb_sklearn": SklearnGaussianNB().fit(train_matrix, y_train_array),
    }

    log.info("evaluating")
    results, predictions = evaluate_all(models, val_matrix, y_val_array)

    if profile == "legacy":
        return {"profile": profile, "leaky": True, "models": results}

    # --- write the bundle ---------------------------------------------------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    object.__setattr__(stats, "demoted_features", tuple(sorted(demoted)))
    write_json(out_dir / "fitted_stats.json", stats_to_json(stats))

    np.savez_compressed(
        out_dir / "models" / "knn_scratch.npz",
        X=knn_reference,
        y=knn_reference_labels,
        k=np.array(knn_k),
    )
    joblib.dump(models["knn_sklearn"], out_dir / "models" / "knn_sklearn.joblib")
    joblib.dump(models["nb_sklearn"], out_dir / "models" / "nb_sklearn.joblib")
    write_json(out_dir / "models" / "nb_scratch.json", nb_scratch_to_json(models["nb_scratch"]))

    write_json(out_dir / "feature_reference_stats.json", _feature_reference_stats(
        train_matrix, y_train_array
    ))
    write_json(out_dir / "dataset_profile.json", _dataset_profile(X, y))

    predictions.to_parquet(out_dir / "eval_predictions.parquet", index=False)

    # A stratified sample, so the dataset explorer never opens the 29 MB CSV.
    sample = pd.concat(
        [
            X.assign(label=y)[y == 0].sample(n=min(4000, int((y == 0).sum())), random_state=7),
            X.assign(label=y)[y == 1].sample(n=min(16000, int((y == 1).sum())), random_state=7),
        ]
    ).sample(frac=1.0, random_state=7)
    sample.to_parquet(out_dir / "dataset_sample.parquet", index=False)

    # The golden row: one raw record with its expected vector and expected predictions.
    # The container re-derives it at start-up, which proves the artifacts that shipped are
    # the artifacts that were tested and that the installed dependencies produce the same
    # numbers.
    golden_raw = X_val.iloc[[0]]
    golden_vector = preprocessor.transform_matrix(golden_raw.copy())[0]
    golden = {
        "raw": json.loads(golden_raw.to_json(orient="records"))[0],
        "vector": [float(v) for v in golden_vector],
        "predictions": {
            key: int(model.predict(golden_vector.reshape(1, -1))[0])
            for key, model in models.items()
        },
        "scores": {
            key: float(model.score_phishing(golden_vector.reshape(1, -1))[0])
            for key, model in models.items()
        },
    }
    write_json(out_dir / "golden_row.json", golden)

    metrics_payload: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "positive_class": "phishing (label 0)",
        "baseline_accuracy": schema.MAJORITY_BASELINE_ACCURACY,
        "baseline_note": (
            "A constant 'legitimate' predictor scores this. Read every accuracy against "
            "it: on its own the number cannot tell a working detector from a constant."
        ),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "profiles": {
            "corrected": {
                "leaky": False,
                "knn_k": knn_k,
                "note": "Canonical. Serving-time statistics are the training-time statistics.",
                "models": results,
            },
        },
    }

    if with_legacy:
        log.info("reproducing the legacy profile for comparison")
        legacy = train("legacy", source, out_dir, demoted, with_legacy=False)
        metrics_payload["profiles"]["legacy"] = {
            "leaky": True,
            "knn_k": KNN_K_LEGACY,
            "note": (
                "Reconstruction of the original's configuration, including standardising "
                "each split by its own mean and standard deviation. Reported for "
                "provenance only. The scaling behind these numbers does not exist at "
                "serving time and cannot be reconstructed from a single URL, so they "
                "describe a model that cannot be deployed."
            ),
            "models": legacy["models"],
        }
    write_json(out_dir / "metrics.json", metrics_payload)

    # The agreement report is an input to training, not a by-product of it: the demotion
    # list decides what the extractor will emit at serving time, and a model fitted on
    # features it will never receive is fitted on the wrong thing.
    if agreement is not None and agreement.exists():
        report = json.loads(agreement.read_text(encoding="utf-8"))
        reported = set(report.get("demoted", ()))
        if reported != set(demoted):
            log.warning(
                "agreement report demotes %s but --demote was given %s; the bundle records "
                "what was actually trained on",
                sorted(reported),
                sorted(demoted),
            )
        report["trained_with_demoted"] = sorted(demoted)
        write_json(out_dir / "extraction_agreement.json", report)
    else:
        write_json(
            out_dir / "extraction_agreement.json",
            {
                "status": "not_run",
                "note": (
                    "The page-feature extraction rules are reconstructions, not the "
                    "dataset's own code. Until the agreement harness has run, no feature "
                    "has been demoted and none has been shown to be trustworthy either."
                ),
                "demoted": sorted(demoted),
            },
        )

    write_json(
        out_dir / "manifest.json",
        build_manifest(
            out_dir,
            {
                "profile": "corrected",
                "knn_k": knn_k,
                "knn_reference_rows": reference_rows,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "demoted_features": sorted(demoted),
            },
        ),
    )

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["corrected", "legacy"], default="corrected")
    parser.add_argument("--source", type=Path, default=TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--demote",
        default="",
        help="comma-separated page features to demote, from the agreement gate",
    )
    parser.add_argument(
        "--agreement",
        type=Path,
        default=None,
        help="extraction agreement report to fold into the bundle",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="skip reproducing the original's leaky configuration for comparison",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )

    demoted = frozenset(f.strip() for f in args.demote.split(",") if f.strip())
    unknown = demoted - set(schema.HTML_FEATURES)
    if unknown:
        raise SystemExit(f"--demote names features that are not page features: {sorted(unknown)}")

    started = time.monotonic()
    payload = train(
        args.profile,
        args.source,
        args.out,
        demoted,
        with_legacy=(args.profile == "corrected" and not args.no_legacy),
        agreement=args.agreement,
    )
    elapsed = time.monotonic() - started

    print(f"\nprofile: {args.profile}  ({elapsed:.1f}s)")
    models = (
        payload["models"] if args.profile == "legacy" else payload["profiles"]["corrected"]["models"]
    )
    print(f"{'model':34} {'acc':>8} {'phish recall':>13} {'phish prec':>11} {'PR-AUC':>8}")
    for key, record in models.items():
        if key.startswith("_"):
            continue
        print(
            f"{record['name']:34} {record['accuracy']:8.5f} {record['phishing_recall']:13.5f} "
            f"{record['phishing_precision']:11.5f} {record['average_precision_phishing']:8.5f}"
        )
    print(f"\nconstant-predictor baseline accuracy: {schema.MAJORITY_BASELINE_ACCURACY}")
    print(f"scratch-vs-reference disagreement: {models['_parity']}")


if __name__ == "__main__":
    main()
