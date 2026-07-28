"""The only module in the package permitted to open a socket.

Redirects are followed manually with ``allow_redirects=False`` so that every hop goes
through the same validation as the first. Letting requests follow them would mean the
first hop is vetted and the rest are not, which is the same as not vetting at all: a
public URL that 302s to 127.0.0.1 would sail straight through.

The size cap is accounted on *decompressed* bytes as they are read, not on the
Content-Length header. A few hundred kilobytes of gzip can expand to gigabytes, and a cap
that trusts the declared length does not notice.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter

from phishguard.config import FETCH, FetchConfig
from phishguard.fetch.safety import (
    FetchOutcome,
    UrlRejected,
    VettedTarget,
    looks_like_challenge,
    resolve_and_vet,
)

log = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

#: Process-wide ceiling on concurrent outbound requests. Together with the byte cap this
#: bounds in-flight memory: concurrency * max_bytes.
_SEMAPHORE = threading.Semaphore(FETCH.concurrency)


@dataclass
class FetchResult:
    """Everything the extractor needs, and nothing that identifies the visitor."""

    outcome: FetchOutcome
    url: str
    final_url: str = ""
    status_code: int | None = None
    html: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    content_type: str = ""
    elapsed_s: float = 0.0
    reason: str = ""
    bytes_read: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome is FetchOutcome.OK

    @property
    def state(self) -> str:
        """The three states the interface renders.

        Deliberately three and not two. "Fetched but challenged" is neither a success nor
        a failure to reach the host, and collapsing it into either one would misdescribe
        what the model was given.
        """
        if self.outcome is FetchOutcome.OK:
            return "scraped"
        if self.outcome is FetchOutcome.CHALLENGE_DETECTED:
            return "challenged"
        return "unreachable"


class _PinnedAdapter(HTTPAdapter):
    """Connect to a pre-vetted IP while preserving the hostname for TLS and Host.

    This is what makes validation more than decorative. Without pinning, the address is
    resolved once for the check and again by urllib3 when the socket opens, and a hostile
    DNS server is free to answer differently the second time. The certificate is still
    verified against the original hostname, so pinning does not weaken TLS.
    """

    def __init__(self, target: VettedTarget, **kwargs: Any) -> None:
        self._target = target
        super().__init__(**kwargs)

    def send(self, request: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        parsed = urlsplit(request.url)
        if parsed.hostname == self._target.host:
            host_header = parsed.netloc
            literal = f"[{self._target.ip}]" if ":" in self._target.ip else self._target.ip
            pinned_netloc = f"{literal}:{self._target.port}"
            request.url = request.url.replace(parsed.netloc, pinned_netloc, 1)
            request.headers["Host"] = host_header
            kwargs.setdefault("verify", True)
        return super().send(request, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # Verify the certificate against the name the user asked for, not the IP we
        # dialled. Without this, pinning would break TLS validation.
        kwargs["server_hostname"] = self._target.host
        kwargs["assert_hostname"] = self._target.host
        super().init_poolmanager(*args, **kwargs)


def _build_session(target: VettedTarget, config: FetchConfig) -> requests.Session:
    session = requests.Session()
    # Ignore ambient proxy environment variables: an operator-set http_proxy would
    # silently route every request somewhere the address vetting never examined.
    session.trust_env = False
    session.max_redirects = 0
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Language": "en",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    if config.egress_proxy:
        session.proxies = {"http": config.egress_proxy, "https": config.egress_proxy}
    else:
        session.mount(f"{target.scheme}://", _PinnedAdapter(target, max_retries=0))
    return session


def _read_capped(response: requests.Response, max_bytes: int) -> tuple[bytes, bool]:
    """Stream the body, stopping once the decompressed size exceeds the cap."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=16384, decode_unicode=False):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            return b"".join(chunks), True
        chunks.append(chunk)
    return b"".join(chunks), False


def get(url: str, config: FetchConfig | None = None) -> FetchResult:
    """Fetch a page under the full guard. Never raises for an expected failure."""
    config = config or FETCH

    if not config.enabled:
        return FetchResult(
            outcome=FetchOutcome.DISABLED,
            url=url,
            reason="fetching is disabled by configuration",
        )

    started = time.monotonic()
    current = url
    redirect_chain: list[str] = []

    with _SEMAPHORE:
        for hop in range(config.max_redirects + 1):
            remaining = config.total_timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                return FetchResult(
                    outcome=FetchOutcome.TIMEOUT,
                    url=url,
                    final_url=current,
                    redirect_chain=redirect_chain,
                    elapsed_s=time.monotonic() - started,
                    reason="wall-clock budget exhausted across redirects",
                )

            # Every hop is revalidated in full. The previous hop having been safe says
            # nothing about this one.
            try:
                target = resolve_and_vet(current, config)
            except UrlRejected as exc:
                return FetchResult(
                    outcome=(
                        FetchOutcome.DNS_FAIL
                        if exc.reason.startswith("dns_")
                        else FetchOutcome.BLOCKED_BY_POLICY
                    ),
                    url=url,
                    final_url=current,
                    redirect_chain=redirect_chain,
                    elapsed_s=time.monotonic() - started,
                    reason=exc.reason,
                )

            session = _build_session(target, config)
            try:
                response = session.get(
                    target.url,
                    timeout=(config.connect_timeout_s, min(config.read_timeout_s, remaining)),
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout:
                return FetchResult(
                    outcome=FetchOutcome.TIMEOUT,
                    url=url,
                    final_url=current,
                    redirect_chain=redirect_chain,
                    elapsed_s=time.monotonic() - started,
                    reason="request timed out",
                )
            except requests.RequestException as exc:
                return FetchResult(
                    outcome=FetchOutcome.HTTP_ERROR,
                    url=url,
                    final_url=current,
                    redirect_chain=redirect_chain,
                    elapsed_s=time.monotonic() - started,
                    reason=type(exc).__name__,
                )

            try:
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location", "")
                    if not location:
                        return _http_error(url, current, redirect_chain, started, response)
                    redirect_chain.append(current)
                    current = urljoin(current, location)
                    if hop == config.max_redirects:
                        return FetchResult(
                            outcome=FetchOutcome.BLOCKED_BY_POLICY,
                            url=url,
                            final_url=current,
                            redirect_chain=redirect_chain,
                            elapsed_s=time.monotonic() - started,
                            reason="too_many_redirects",
                        )
                    continue

                if response.status_code >= 400:
                    return _http_error(url, current, redirect_chain, started, response)

                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type and not any(
                    content_type == allowed for allowed in ALLOWED_CONTENT_TYPES
                ):
                    return FetchResult(
                        outcome=FetchOutcome.UNSUPPORTED_TYPE,
                        url=url,
                        final_url=current,
                        status_code=response.status_code,
                        redirect_chain=redirect_chain,
                        content_type=content_type,
                        elapsed_s=time.monotonic() - started,
                        reason=f"content type {content_type!r} is not HTML",
                    )

                body, truncated = _read_capped(response, config.max_bytes)
                if truncated:
                    return FetchResult(
                        outcome=FetchOutcome.TOO_LARGE,
                        url=url,
                        final_url=current,
                        status_code=response.status_code,
                        redirect_chain=redirect_chain,
                        content_type=content_type,
                        elapsed_s=time.monotonic() - started,
                        bytes_read=len(body),
                        reason=f"body exceeded {config.max_bytes} decompressed bytes",
                    )

                encoding = response.encoding or response.apparent_encoding or "utf-8"
                html = body.decode(encoding, errors="replace")

                outcome = (
                    FetchOutcome.CHALLENGE_DETECTED
                    if looks_like_challenge(html, None)
                    else FetchOutcome.OK
                )
                return FetchResult(
                    outcome=outcome,
                    url=url,
                    final_url=current,
                    status_code=response.status_code,
                    html=html,
                    redirect_chain=redirect_chain,
                    content_type=content_type,
                    elapsed_s=time.monotonic() - started,
                    bytes_read=len(body),
                    reason="bot challenge or interstitial" if outcome is not FetchOutcome.OK else "",
                )
            finally:
                response.close()
                session.close()

    return FetchResult(
        outcome=FetchOutcome.BLOCKED_BY_POLICY,
        url=url,
        final_url=current,
        redirect_chain=redirect_chain,
        elapsed_s=time.monotonic() - started,
        reason="too_many_redirects",
    )


def _http_error(
    url: str,
    current: str,
    chain: list[str],
    started: float,
    response: requests.Response,
) -> FetchResult:
    return FetchResult(
        outcome=FetchOutcome.HTTP_ERROR,
        url=url,
        final_url=current,
        status_code=response.status_code,
        redirect_chain=chain,
        elapsed_s=time.monotonic() - started,
        reason=f"HTTP {response.status_code}",
    )


def get_robots(url: str, config: FetchConfig | None = None) -> bool | None:
    """Fetch /robots.txt under the same guard.

    Returns True when crawling is permitted, False on a blanket disallow, None when the
    file could not be retrieved. This is a *feature input*, not an authorisation check --
    a disallow does not stop the page fetch, it only sets the Robots feature.
    """
    config = config or FETCH
    if not config.enabled or not config.robots:
        return None

    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None

    result = get(f"{parts.scheme}://{parts.netloc}/robots.txt", config)
    if not result.ok or result.status_code != 200:
        return None

    return not _has_blanket_disallow(result.html)


def _has_blanket_disallow(robots_txt: str) -> bool:
    """True when the ``User-agent: *`` group contains a bare ``Disallow: /``."""
    in_star_group = False
    for raw_line in robots_txt.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, _, value = line.partition(":")
        field_name = field_name.strip().lower()
        value = value.strip()
        if field_name == "user-agent":
            in_star_group = value == "*"
        elif field_name == "disallow" and in_star_group and value == "/":
            return True
    return False
