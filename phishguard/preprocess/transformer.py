"""The fit/transform split.

``fit`` learns; ``transform`` only reads. There is no ``transform_single``, no
``predict_one`` shortcut, and no "skip clipping for a single row" fast path. Every such
special case is a place where the serving path can drift from the training path, and drift
of exactly that kind is the defect this module exists to eliminate.

The proof obligation is a test, not a comment: transforming n rows individually must equal
transforming them as one batch, bitwise on float32. Any tolerance-based comparison would
pass even if a statistic were being recomputed from a batch that happens to resemble the
training set. Exact equality can only hold if transform reads its numbers from FittedStats.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from phishguard import schema
from phishguard.features import url_features as uf
from phishguard.preprocess.scaler import SubsetStandardScaler
from phishguard.preprocess.stats import (
    CategoricalFillStep,
    ClipBound,
    FallbackSource,
    FillMethod,
    FittedStats,
    HtmlFallback,
    NbDropReason,
    NumericFill,
)


class FeatureContractError(RuntimeError):
    """A feature column is absent at reindex time.

    Distinct from a missing *value*, which is normal and is imputed. An absent column
    means the extractor's contract broke, and failing loudly is correct.
    """


class NotFittedError(RuntimeError):
    pass


def _apply_url_fill_chain(
    X: pd.DataFrame,
    *,
    char_prob: dict[str, float],
    tld_prob_mean: pd.Series,
    tld_prob_global_fill: float,
) -> pd.DataFrame:
    """The 26 fill_* functions, in the one order their dependencies allow.

    Domain must precede DomainLength, TLD and NoOfSubDomain; TLD must precede TLDLength
    and TLDLegitimateProb; URLLength must precede every ratio that divides by it;
    IsDomainIP must precede TLDLegitimateProb.

    Every function here is row-local given the three injected statistics, which is the
    property the invariance test depends on.
    """
    X = uf.fill_url_length(X, url_col="URL", url_length_col="URLLength")
    X = uf.fill_domain(X, url_col="URL", domain_col="Domain")
    X = uf.fill_domain_length(X, domain_col="Domain", domain_length_col="DomainLength")
    X = uf.fill_is_domain_ip(X, domain_col="Domain", is_domain_ip_col="IsDomainIP")
    X = uf.fill_tld(X, domain_col="Domain", tld_col="TLD")
    X = uf.fill_char_continuation_rate(X, url_col="URL", char_rate_col="CharContinuationRate")
    X = uf.fill_tld_legitimate_prob(
        X,
        tld_prob_mean=tld_prob_mean,
        global_fill_value=tld_prob_global_fill,
        tld_col="TLD",
        tld_prob_col="TLDLegitimateProb",
        is_domain_ip_col="IsDomainIP",
    )
    X = uf.fill_url_char_prob(
        X, url_col="URL", char_prob_col="URLCharProb", char_prob=char_prob
    )
    X = uf.fill_tld_length(X, tld_col="TLD", tld_length_col="TLDLength")
    X = uf.fill_no_of_subdomains(X, domain_col="Domain", subdomain_col="NoOfSubDomain")
    X = uf.fill_has_obfuscation(X, url_col="URL", obfuscation_col="HasObfuscation")
    X = uf.fill_no_of_obfuscated_characters(X, url_col="URL", obf_char_col="NoOfObfuscatedChar")
    X = uf.fill_obfuscation_ratio(
        X,
        obf_char_col="NoOfObfuscatedChar",
        url_length_col="URLLength",
        obf_ratio_col="ObfuscationRatio",
    )
    X = uf.fill_no_of_letters_in_url(X, url_col="URL", letters_col="NoOfLettersInURL")
    X = uf.fill_letter_ratio_in_url(
        X,
        letters_col="NoOfLettersInURL",
        url_length_col="URLLength",
        ratio_col="LetterRatioInURL",
    )
    X = uf.fill_no_of_digits_in_url(X, url_col="URL", digits_col="NoOfDegitsInURL")
    X = uf.fill_digits_ratio_in_url(X, url_col="URL", ratio_col="DegitRatioInURL")
    X = uf.fill_no_of_equals_in_url(X, url_col="URL", equals_col="NoOfEqualsInURL")
    X = uf.fill_no_of_qmark_in_url(X, url_col="URL", qmark_col="NoOfQMarkInURL")
    X = uf.fill_no_of_ampersand_in_url(X, url_col="URL", ampersand_col="NoOfAmpersandInURL")
    X = uf.fill_no_of_special_chars_in_url(
        X, url_col="URL", special_chars_col="NoOfOtherSpecialCharsInURL"
    )
    X = uf.fill_specials_ratio_in_url(X, url_col="URL", ratio_col="SpacialCharRatioInURL")
    X = uf.fill_is_https(X, url_col="URL", https_col="IsHTTPS")
    X = uf.fill_has_title(X, title_col="Title", has_title_col="HasTitle")
    X = uf.fill_domain_title_match_score(
        X, title_col="Title", domain_col="Domain", score_col="DomainTitleMatchScore"
    )
    return uf.fill_url_title_match_score(
        X, title_col="Title", url_col="URL", score_col="URLTitleMatchScore"
    )


def _group_mode(values: pd.Series) -> Any:
    """Most frequent value in a group, or None when the group is entirely missing."""
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else None


def _cascade_key(row: Any, groupby_cols: tuple[str, ...]) -> tuple[float, ...]:
    """Build the cascade lookup key for one row.

    Values are coerced to float so that a key built from an int column and a key built
    from the same column after a float round-trip compare equal. Without this, a one-row
    frame (whose columns pandas may infer as int64) and a batch frame (float64 because
    some other row had a NaN) would miss each other's mode tables, and invariance would
    fail for reasons that have nothing to do with the statistics.
    """
    return tuple(float(row[c]) for c in groupby_cols)


class Preprocessor:
    """Learns from the training split, then applies what it learned and nothing else."""

    def __init__(self, stats: FittedStats | None = None) -> None:
        self._stats = stats
        #: Per-cascade-step count of rows that fell back to the global mode. Instrumented
        #: because a step with a high fallback rate is telling you its conditioning is
        #: doing nothing useful -- worth knowing, not an error.
        self.fallback_counts: Counter[str] = Counter()

    @property
    def stats(self) -> FittedStats:
        if self._stats is None:
            raise NotFittedError("Preprocessor.transform called before fit(); no statistics loaded")
        return self._stats

    @property
    def is_fitted(self) -> bool:
        return self._stats is not None

    # -- fit ---------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FittedStats:
        """Compute every statistic the serving path will need.

        Returns FittedStats and nothing else -- not a transformed matrix. The caller
        invokes transform(X_train) afterwards. That duplicates steps 1 and 2 and is worth
        it: it makes "the training matrix is produced by the same code path as a single
        live URL" true by construction rather than by inspection.
        """
        X = X_train.copy()

        # 1. Character frequency table over the training corpus' URLs.
        char_prob = uf.calculate_char_prob(X, "URL")

        # 2. Materialise the string-derived columns so later steps see a filled TLD to
        #    group by. This is a dry run of transform's first stage.
        tld_prob_mean_series, tld_global_fill = uf.compute_tld_prob_statistics(X)
        tld_skew = uf.scalar_float(X["TLDLegitimateProb"].skew())
        tld_fill_method: FillMethod = "median" if abs(tld_skew) > 1 else "mean"

        X = _apply_url_fill_chain(
            X,
            char_prob=char_prob,
            tld_prob_mean=tld_prob_mean_series,
            tld_prob_global_fill=tld_global_fill,
        )

        # 3/4. Numeric imputation. The threshold is 3 here and 1 for the TLD statistic
        #      above. The two are genuinely different and are not unified.
        numeric_fill: list[NumericFill] = []
        for col in schema.NUMERICAL_COLUMNS:
            skewness = uf.scalar_float(X[col].skew())
            method: FillMethod
            if abs(skewness) > 3:
                method = "median"
                value = uf.scalar_float(X[col].median())
            else:
                method = "mean"
                value = uf.scalar_float(X[col].mean())
            numeric_fill.append(
                NumericFill(column=col, method=method, value=value, skew=skewness)
            )
            X[col] = X[col].fillna(value)

        # 5. The ordered categorical cascade.
        cascade = self._fit_cascade(X)
        for step in cascade:
            X[step.column] = self._fill_cascade_step(X, step, count_fallbacks=False)

        # 6. Clip bounds, computed after imputation so the quantiles are over the filled
        #    distribution.
        clip_bounds = tuple(
            ClipBound(
                column=col,
                lower=float(np.percentile(X[col].dropna(), 1.0)),
                upper=float(np.percentile(X[col].dropna(), 99.0)),
            )
            for col in schema.NUMERICAL_COLUMNS
        )
        for b in clip_bounds:
            X[b.column] = np.clip(X[b.column], b.lower, b.upper)

        # 7. Scaler over the 30 numerics only.
        X_features = X.drop(columns=list(schema.INTERMEDIATE_COLUMNS))
        X_features = _validate_dtypes(X_features)
        X_features = X_features[list(schema.FEATURE_ORDER)]
        scaler_params = SubsetStandardScaler(schema.NUMERICAL_COLUMNS).fit(X_features)
        X_scaled = SubsetStandardScaler.apply(X_features, scaler_params)

        # 8. Naive-Bayes column drop, last because it needs the finished matrix.
        nb_drop, nb_drop_reasons = naive_bayes_drop(X_scaled, y_train)

        # 9. Page-feature fallbacks, derived from the fills computed above rather than
        #    computed again. Lifted out so the interface can display what it fell back to.
        numeric_map = {f.column: f for f in numeric_fill}
        cascade_map = {s.column: s for s in cascade}
        html_fallbacks: list[HtmlFallback] = []
        for col in (*schema.HTML_FEATURES, *schema.TITLE_HYBRID_FEATURES):
            if col in numeric_map:
                nf = numeric_map[col]
                source: FallbackSource = (
                    "numeric_median" if nf.method == "median" else "numeric_mean"
                )
                html_fallbacks.append(
                    HtmlFallback(column=col, value=nf.value, source=source)
                )
            elif col in cascade_map:
                html_fallbacks.append(
                    HtmlFallback(
                        column=col,
                        value=float(cascade_map[col].global_mode),
                        source="categorical_mode",
                    )
                )

        stats = FittedStats(
            char_prob=char_prob,
            tld_prob_mean={str(k): float(v) for k, v in tld_prob_mean_series.items()},
            tld_prob_global_fill=float(tld_global_fill),
            tld_skew=tld_skew,
            tld_fill_method=tld_fill_method,
            numeric_fill=tuple(numeric_fill),
            categorical_cascade=cascade,
            clip_bounds=clip_bounds,
            scaler=scaler_params,
            nb_drop=nb_drop,
            nb_drop_reasons=nb_drop_reasons,
            html_fallbacks=tuple(html_fallbacks),
            feature_order=schema.FEATURE_ORDER,
            numerical_columns=schema.NUMERICAL_COLUMNS,
            continuous_columns=schema.CONTINUOUS_COLUMNS,
            discrete_columns=schema.DISCRETE_COLUMNS,
            categorical_columns_filtered=schema.CATEGORICAL_COLUMNS_FILTERED,
            n_train_rows=int(len(X_train)),
        )
        self._stats = stats
        return stats

    def _fit_cascade(self, X: pd.DataFrame) -> tuple[CategoricalFillStep, ...]:
        """Learn the 18 mode tables, growing the group key one column per step.

        The key grows monotonically: step 1 groups by HasObfuscation alone, step 2 by
        HasObfuscation and the column step 1 just filled, and so on. So column n is
        imputed conditioned on columns 1..n-1, and permuting the list changes every mode
        table from step 2 onward. It is not a cosmetic ordering.
        """
        steps: list[CategoricalFillStep] = []
        groupby_cols: list[str] = list(schema.CATEGORICAL_FILL_INITIAL_GROUP_BY)

        for col in schema.CATEGORICAL_COLUMNS_TO_FILL:
            grouped = X.groupby(groupby_cols, dropna=False)[col].agg(_group_mode)
            col_mode = X[col].mode()
            global_mode = uf.scalar_float(col_mode.iloc[0]) if not col_mode.empty else 0.0

            modes: dict[tuple[float, ...], float] = {}
            for key, value in grouped.items():
                if value is None or pd.isna(value):
                    continue
                key_tuple = key if isinstance(key, tuple) else (key,)
                if any(pd.isna(k) for k in key_tuple):
                    continue
                modes[tuple(float(k) for k in key_tuple)] = float(value)

            step = CategoricalFillStep(
                column=col,
                groupby_cols=tuple(groupby_cols),
                modes=modes,
                global_mode=global_mode,
            )
            steps.append(step)
            X[col] = self._fill_cascade_step(X, step, count_fallbacks=False)
            groupby_cols.append(col)

        return tuple(steps)

    # -- transform ---------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted statistics. Computes nothing.

        Accepts a frame in the raw schema shape -- the same shape a row of the training
        CSV has -- whether it holds 28,081 rows or 1. There is no branch on len(X).
        """
        stats = self.stats

        # Validate the input contract before touching anything. A missing feature column
        # is a broken extractor, not an unknown value, and it must be reported by name
        # rather than surfacing later as a bare KeyError from whichever stage happened to
        # reach for it first.
        required = {*stats.feature_order, *schema.INTERMEDIATE_COLUMNS}
        missing = sorted(required - set(X.columns))
        if missing:
            raise FeatureContractError(f"input columns absent from frame: {missing}")

        X = X.copy()

        tld_prob_mean = pd.Series(stats.tld_prob_mean, dtype="float64")

        # 1. The 26 row-local fill functions, with the three fitted statistics injected.
        X = _apply_url_fill_chain(
            X,
            char_prob=stats.char_prob,
            tld_prob_mean=tld_prob_mean,
            tld_prob_global_fill=stats.tld_prob_global_fill,
        )

        # 2. Numeric imputation from recorded values.
        for f in stats.numeric_fill:
            X[f.column] = X[f.column].fillna(f.value)

        # 3. Replay the cascade in recorded order.
        for step in stats.categorical_cascade:
            X[step.column] = self._fill_cascade_step(X, step, count_fallbacks=True)

        # 4. Clip to recorded bounds.
        for b in stats.clip_bounds:
            X[b.column] = np.clip(X[b.column], b.lower, b.upper)

        # 5. Drop the text intermediates.
        X = X.drop(columns=[c for c in schema.INTERMEDIATE_COLUMNS if c in X.columns])

        # 6. Reindex to the frozen order. A missing column is a broken contract, not an
        #    unknown value, so this raises rather than filling.
        missing = sorted(set(stats.feature_order) - set(X.columns))
        if missing:
            raise FeatureContractError(f"feature columns absent from frame: {missing}")
        X = X[list(stats.feature_order)]

        # 7. Dtype validation, with the column lists passed in explicitly rather than
        #    closed over. A helper that closes over a module global is a train/serve skew
        #    bug waiting for someone to import the module in a different order.
        X = _validate_dtypes(
            X,
            binary_columns=(*stats.categorical_columns_filtered, *stats.discrete_columns),
        )

        # 8. Scale the 30 numerics.
        X = SubsetStandardScaler.apply(X, stats.scaler)

        return X

    def transform_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """transform(), fixed to float32 at the boundary.

        The dtype is pinned here so the KNN distance kernel never silently upcasts and
        never sees a dtype it did not see during training.
        """
        return self.transform(X).to_numpy(dtype=np.float32)

    def _fill_cascade_step(
        self, X: pd.DataFrame, step: CategoricalFillStep, *, count_fallbacks: bool
    ) -> pd.Series:
        """Fill one cascade column from its recorded mode table.

        An unseen group key is expected, not exceptional. At step 18 the key is 18 columns
        wide, so the training frame observed only a small fraction of the combinatorial
        space. The defined behaviour is to fall back to the recorded global mode -- not to
        raise, not to warn at the user, and under no circumstances to recompute the mode
        from the incoming batch, which would reintroduce exactly the defect this module
        exists to remove.
        """
        col = X[step.column]
        null_mask = col.isna()
        if not null_mask.any():
            return col

        modes = step.modes
        filled = col.copy()
        for idx in col.index[null_mask]:
            row = X.loc[idx]
            try:
                key = _cascade_key(row, step.groupby_cols)
            except (TypeError, ValueError):
                key = None
            value = modes.get(key, step.global_mode) if key is not None else step.global_mode
            if count_fallbacks and (key is None or key not in modes):
                self.fallback_counts[step.column] += 1
            filled.at[idx] = value
        return filled


def _validate_dtypes(
    X: pd.DataFrame, binary_columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Coerce the integral columns to int, as the original did.

    Column lists arrive as a parameter. The original closed over notebook globals here,
    which is how the scaler silently ended up touching only 30 of 49 columns.
    """
    if binary_columns is None:
        binary_columns = (*schema.CATEGORICAL_COLUMNS_FILTERED, *schema.DISCRETE_COLUMNS)

    X = X.copy()
    cols = [c for c in binary_columns if c in X.columns]
    if cols:
        X[cols] = X[cols].astype("int64")
    return X


def naive_bayes_drop(
    X: pd.DataFrame, y: pd.Series
) -> tuple[tuple[str, ...], dict[str, NbDropReason]]:
    """Columns Gaussian NB cannot use, with the reason recorded per column.

    Two rules, ported from the original:

      - a binary indicator with an empty (value, label) contingency cell, which would give
        that cell zero likelihood;
      - any column with zero standard deviation, whose Gaussian is degenerate.

    The original computed this list and discarded it, so nothing downstream could say
    which columns a served model had actually been fitted on. Here it is returned,
    deduplicated with the first reason winning, and persisted.
    """
    nb_df = X.join(y.rename(schema.TARGET_COLUMN))
    reasons: dict[str, NbDropReason] = {}
    ordered: list[str] = []

    for f in schema.CATEGORICAL_COLUMNS_FILTERED:
        if f not in X.columns:
            continue
        for value, label in [(x // 2, x % 2) for x in range(4)]:
            empty = len(nb_df[(nb_df[f] == value) & (nb_df[schema.TARGET_COLUMN] == label)]) == 0
            if empty and f not in reasons:
                reasons[f] = "empty_contingency_cell"
                ordered.append(f)

    for col in X.columns:
        if uf.scalar_float(X[col].std()) == 0.0 and col not in reasons:
            reasons[col] = "zero_std"
            ordered.append(col)

    return tuple(ordered), reasons
