"""URL validation and address vetting for outbound requests.

THREAT MODEL
============

The service fetches a URL supplied by an anonymous visitor. That is a server-side request
forgery primitive by construction: whatever the visitor types, the server connects to. The
controls here exist to make sure it can only ever reach a public internet host.

What is defended against:

  - Direct requests to loopback, private, link-local, carrier-grade NAT, multicast and
    reserved ranges -- including the cloud metadata address at 169.254.169.254, which on
    most providers hands out credentials to anyone who asks.
  - Requests to the application's own public address, which would otherwise let a visitor
    aim the fetcher at the service itself.
  - Hostnames that *resolve* into those ranges. A name is not safe because it looks
    public: `localtest.me` resolves to 127.0.0.1 and is a perfectly ordinary public DNS
    record. Only the resolved address can be judged.
  - DNS rebinding, where the name resolves to a public address during validation and to a
    private one when the socket is actually opened. Validating the name and then letting
    the HTTP stack resolve it again independently is the classic mistake, and it makes the
    validation decorative. The address that passed validation is the address connected to
    -- see the pinning adapter in client.py.
  - Redirects into any of the above. Each hop is revalidated in full rather than trusted
    because the first hop was clean.
  - Non-HTTP schemes (file://, gopher://, ftp://) and non-standard ports.
  - Credentials embedded in the URL, which can be used to confuse downstream parsers.

What is deliberately not defended against: a hostile *public* site serving hostile
content. That is what the transfer caps, the timeouts, the content-type allowlist and the
absence of any JavaScript execution are for.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from phishguard.config import FetchConfig

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
DEFAULT_PORTS = {"http": 80, "https": 443}


class FetchOutcome(str, Enum):
    """Why a fetch ended the way it did.

    CHALLENGED matters more than it looks. The dataset was crawled in 2023-24, so
    re-fetching its URLs today mostly yields dead domains, parked pages and CDN
    interstitials. A bot wall returning HTTP 200 is not a successful scrape: its HTML
    describes the challenge, not the site. Folding it into OK would feed the model
    features of the interstitial and report them as real evidence.
    """

    OK = "ok"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    DNS_FAIL = "dns_fail"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    HTTP_ERROR = "http_error"
    CHALLENGE_DETECTED = "challenge_detected"
    DISABLED = "disabled"
    RATE_LIMITED = "rate_limited"


class UrlRejected(ValueError):
    """The URL failed policy validation. Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class VettedTarget:
    """A URL that passed validation, together with the address it may connect to."""

    url: str
    scheme: str
    host: str
    port: int
    ip: str

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def _is_forbidden_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string when the address is not a permissible destination."""
    # Order matters only for the reported reason, not for the decision -- every branch
    # here blocks. The narrower checks come first so the log says what was actually wrong:
    # 0.0.0.0 and the documentation ranges are all inside is_private, which would
    # otherwise swallow them under a less useful label.
    if ip.is_unspecified:
        return "unspecified_address"
    if ip.is_loopback:
        return "loopback_address"
    if ip.is_private:
        # Covers RFC1918, unique-local IPv6 and the 169.254/16 link-local block, which is
        # where the cloud metadata service lives.
        return "private_address"
    if ip.is_link_local:
        return "link_local_address"
    if ip.is_multicast:
        return "multicast_address"
    if ip.is_reserved:
        return "reserved_address"

    if isinstance(ip, ipaddress.IPv4Address):
        # Carrier-grade NAT. Not covered by is_private, and routable to other tenants.
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return "cgnat_address"
        if ip in ipaddress.ip_network("192.0.0.0/24"):
            return "ietf_protocol_assignment"
    else:
        # An IPv4-mapped IPv6 address would otherwise sidestep every IPv4 rule above.
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            return _is_forbidden_address(mapped) or "ipv4_mapped"
        if ip.is_site_local:
            return "site_local_address"

    return None


def validate_url(raw_url: str, config: FetchConfig) -> tuple[str, str, int]:
    """Check everything decidable from the URL string alone.

    Returns (normalised_url, host, port). Raises UrlRejected otherwise.
    """
    if not raw_url or not raw_url.strip():
        return _reject("empty_url", "")

    url = raw_url.strip()
    if len(url) > 2048:
        return _reject("url_too_long", f"{len(url)} characters")

    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return _reject("unparseable_url", str(exc))

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return _reject("scheme_not_allowed", scheme or "(none)")

    # Credentials in the URL can be used to confuse parsers into disagreeing about which
    # host is being addressed, so they are refused outright rather than stripped.
    if parts.username or parts.password or "@" in (parts.netloc.split("]")[-1]):
        return _reject("credentials_in_url", "")

    hostname = parts.hostname
    if not hostname:
        return _reject("missing_host", "")

    try:
        # IDNA-normalise so that a visually-confusable or mixed-encoding hostname is
        # judged in the same form the socket layer will use.
        host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        try:
            host = hostname.encode("ascii").decode("ascii").lower()
        except UnicodeError:
            return _reject("invalid_hostname", hostname)

    try:
        port = parts.port if parts.port is not None else DEFAULT_PORTS[scheme]
    except ValueError as exc:
        return _reject("invalid_port", str(exc))

    if port not in ALLOWED_PORTS:
        return _reject("port_not_allowed", str(port))

    if host in config.deny_hosts:
        return _reject("host_denied", host)
    # A blocklist entry also covers everything beneath it, so blocking one abusive host
    # does not require enumerating its subdomains.
    if any(host.endswith("." + denied) for denied in config.deny_hosts):
        return _reject("host_denied", host)

    return url, host, port


def _reject(reason: str, detail: str) -> tuple[str, str, int]:
    raise UrlRejected(reason, detail)


def resolve_and_vet(raw_url: str, config: FetchConfig) -> VettedTarget:
    """Validate the URL, resolve the host, and reject unless *every* address is public.

    Every address, not the first: a name that resolves to both a public and a private
    address would otherwise be reachable by whichever the connection happened to pick.
    """
    url, host, port = validate_url(raw_url, config)

    # A literal address needs no DNS, but still needs vetting.
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None

    if literal is not None:
        if str(literal) in config.self_ips:
            raise UrlRejected("own_address", host)
        reason = _is_forbidden_address(literal)
        if reason:
            raise UrlRejected(reason, host)
        return VettedTarget(url=url, scheme=urlsplit(url).scheme.lower(), host=host,
                            port=port, ip=str(literal))

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejected("dns_resolution_failed", f"{host}: {exc}") from exc

    if not infos:
        raise UrlRejected("dns_no_records", host)

    addresses = []
    for info in infos:
        sockaddr = info[4]
        addr = ipaddress.ip_address(sockaddr[0])
        if str(addr) in config.self_ips:
            raise UrlRejected("own_address", f"{host} -> {addr}")
        reason = _is_forbidden_address(addr)
        if reason:
            raise UrlRejected(reason, f"{host} -> {addr}")
        addresses.append(addr)

    # The first vetted address is the one that will be connected to. Returning it, rather
    # than the hostname, is what closes the rebinding window: the socket is opened against
    # this exact address instead of resolving the name a second time.
    return VettedTarget(
        url=url,
        scheme=urlsplit(url).scheme.lower(),
        host=host,
        port=port,
        ip=str(addresses[0]),
    )


def looks_like_challenge(html: str, title: str | None) -> bool:
    """Detect a bot wall or interstitial served with a 200 status."""
    from phishguard.constants import CHALLENGE_MARKERS

    haystack = f"{title or ''} {html[:4096]}".lower()
    return any(marker in haystack for marker in CHALLENGE_MARKERS)
