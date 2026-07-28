"""Runtime configuration, read from the environment once at import.

Every setting fails fast on a bad value. A service that boots with a silently-defaulted
security control is worse than one that refuses to boot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RefScope = Literal["anchor", "all_resources"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_set(name: str) -> frozenset[str]:
    raw = os.environ.get(name, "")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class FetchConfig:
    """Outbound-request policy.

    Raising MAX_REDIRECTS or MAX_BYTES is security-relevant and should not be done
    casually: the in-flight memory bound is CONCURRENCY * MAX_BYTES, which is 16 MiB at
    these defaults and is what keeps a handful of large responses from pushing the
    container into its memory limit.
    """

    #: Master kill switch. False degrades the service to URL-only features, which is a
    #: rendered state rather than an error, and needs no redeploy to flip.
    enabled: bool = True

    #: Whether the Robots feature costs a second guarded request to /robots.txt. When
    #: false it falls back to the weaker <meta name="robots"> proxy. This is never
    #: consulted to decide whether the page fetch is permitted -- robots.txt is a feature
    #: input here, not an authorisation mechanism.
    robots: bool = True

    #: This host's own public addresses. Empty leaves an SSRF hole open to the app itself,
    #: so deployments must set it.
    self_ips: frozenset[str] = field(default_factory=frozenset)

    #: Operational blocklist, for responding to an abuse report without a redeploy.
    deny_hosts: frozenset[str] = field(default_factory=frozenset)

    connect_timeout_s: float = 3.0
    read_timeout_s: float = 5.0
    #: Hard wall-clock deadline across all redirect hops combined.
    total_timeout_s: float = 10.0
    #: Counted on decompressed bytes, so a compression bomb is capped by what it expands
    #: to rather than by what arrives on the wire.
    max_bytes: int = 2 * 1024 * 1024
    max_redirects: int = 3

    concurrency: int = 8
    rate_per_session: int = 10
    rate_window_s: int = 300
    batch_max_urls: int = 25

    egress_proxy: str = ""
    user_agent: str = (
        "phishguard/1.0 (educational URL classifier; +https://github.com/fetiai)"
    )

    @classmethod
    def from_env(cls) -> FetchConfig:
        return cls(
            enabled=_env_bool("FETCH_ENABLED", True),
            robots=_env_bool("FETCH_ROBOTS", True),
            self_ips=_env_set("FETCH_SELF_IPS"),
            deny_hosts=_env_set("FETCH_DENY_HOSTS"),
            connect_timeout_s=_env_float("FETCH_CONNECT_TIMEOUT_S", 3.0, minimum=0.1),
            read_timeout_s=_env_float("FETCH_READ_TIMEOUT_S", 5.0, minimum=0.1),
            total_timeout_s=_env_float("FETCH_TIMEOUT_S", 10.0, minimum=0.5),
            max_bytes=_env_int("FETCH_MAX_BYTES", 2 * 1024 * 1024, minimum=1024),
            max_redirects=_env_int("FETCH_MAX_REDIRECTS", 3, minimum=0, maximum=10),
            concurrency=_env_int("FETCH_CONCURRENCY", 8, minimum=1, maximum=64),
            rate_per_session=_env_int("FETCH_RATE_PER_SESSION", 10, minimum=1),
            rate_window_s=_env_int("FETCH_RATE_WINDOW_S", 300, minimum=1),
            batch_max_urls=_env_int("FETCH_BATCH_MAX_URLS", 25, minimum=1, maximum=500),
            egress_proxy=os.environ.get("EGRESS_PROXY", ""),
        )


@dataclass(frozen=True)
class AppConfig:
    artifacts_dir: Path = Path("artifacts/v1")

    #: Which definition the three reference-count features use. Settled by measured
    #: agreement against the dataset's own values rather than by assertion, because there
    #: is no prior art for these features anywhere in the source material.
    ref_scope: RefScope = "anchor"

    #: Below this share of the 25 page features actually scraped, the service abstains
    #: instead of predicting. Imputation fills from majority-class statistics, which
    #: biases toward "legitimate" -- the exact wrong direction for a phishing detector --
    #: so a confident answer from mostly-imputed evidence would be actively misleading.
    coverage_min_ratio: float = 0.60

    batch_max_rows: int = 5000

    @classmethod
    def from_env(cls) -> AppConfig:
        scope = os.environ.get("REF_SCOPE", "anchor").strip().lower()
        if scope not in {"anchor", "all_resources"}:
            raise ValueError(f"REF_SCOPE must be 'anchor' or 'all_resources', got {scope!r}")
        return cls(
            artifacts_dir=Path(os.environ.get("ARTIFACTS_DIR", "artifacts/v1")),
            ref_scope=scope,  # type: ignore[arg-type]
            coverage_min_ratio=_env_float("COVERAGE_MIN_RATIO", 0.60, minimum=0.0),
            batch_max_rows=_env_int("BATCH_MAX_ROWS", 5000, minimum=1),
        )


FETCH = FetchConfig.from_env()
APP = AppConfig.from_env()
