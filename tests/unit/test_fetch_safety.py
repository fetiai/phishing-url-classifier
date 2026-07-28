"""Tier 7 -- the outbound request guard.

The service fetches a URL an anonymous visitor typed, which is a server-side request
forgery primitive by construction. These tests are the evidence that it can only reach a
public internet host.

No test here touches the network. DNS is stubbed so that the *resolution* cases -- a
public-looking name that resolves somewhere private -- can be tested deterministically,
which is exactly where a string-based check would pass and a real one must not.
"""

from __future__ import annotations

import socket

import pytest

from phishguard.config import FetchConfig
from phishguard.fetch import client, safety
from phishguard.fetch.safety import FetchOutcome, UrlRejected, resolve_and_vet, validate_url

CONFIG = FetchConfig(self_ips=frozenset({"203.0.113.10"}), deny_hosts=frozenset({"blocked.test"}))


@pytest.fixture
def stub_dns(monkeypatch):
    """Point every hostname at an address of the test's choosing."""

    def _install(mapping: dict[str, str]):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host not in mapping:
                raise socket.gaierror(-2, "Name or service not known")
            addr = mapping[host]
            family = socket.AF_INET6 if ":" in addr else socket.AF_INET
            return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, port))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    return _install


# --- string-level rejections -------------------------------------------------

REJECTED_URLS = [
    ("file:///etc/passwd", "scheme_not_allowed"),
    ("gopher://example.com/", "scheme_not_allowed"),
    ("ftp://example.com/x", "scheme_not_allowed"),
    ("javascript:alert(1)", "scheme_not_allowed"),
    ("", "empty_url"),
    ("http://", "missing_host"),
    ("http://user:pw@example.com/", "credentials_in_url"),
    ("http://example.com@evil.test/", "credentials_in_url"),
    ("http://example.com:22/", "port_not_allowed"),
    ("http://example.com:8080/", "port_not_allowed"),
    ("http://blocked.test/", "host_denied"),
    ("http://deep.sub.blocked.test/", "host_denied"),
]


@pytest.mark.parametrize("url,reason", REJECTED_URLS, ids=[r[1] + ":" + r[0][:28] for r in REJECTED_URLS])
def test_url_is_rejected_before_any_lookup(url, reason):
    with pytest.raises(UrlRejected) as exc:
        validate_url(url, CONFIG)
    assert exc.value.reason == reason


@pytest.mark.parametrize(
    "url", ["http://example.com/", "https://example.com:443/a?b=c", "http://example.com:80/"]
)
def test_ordinary_urls_pass_string_validation(url):
    normalised, host, port = validate_url(url, CONFIG)
    assert host == "example.com"
    assert port in (80, 443)


# --- address-level rejections ------------------------------------------------

FORBIDDEN_LITERALS = [
    ("http://127.0.0.1/", "loopback_address"),
    ("http://[::1]/", "loopback_address"),
    ("http://10.0.0.5/", "private_address"),
    ("http://192.168.1.1/", "private_address"),
    ("http://172.16.0.1/", "private_address"),
    ("http://169.254.169.254/latest/meta-data/", "private_address"),
    ("http://100.64.0.1/", "cgnat_address"),
    ("http://0.0.0.0/", "unspecified_address"),
    ("http://224.0.0.1/", "multicast_address"),
    ("http://[::ffff:127.0.0.1]/", "loopback_address"),
]


@pytest.mark.parametrize("url,reason", FORBIDDEN_LITERALS, ids=[c[0] for c in FORBIDDEN_LITERALS])
def test_forbidden_literal_addresses_are_blocked(url, reason):
    with pytest.raises(UrlRejected) as exc:
        resolve_and_vet(url, CONFIG)
    assert exc.value.reason == reason


def test_cloud_metadata_address_is_blocked():
    """169.254.169.254 hands out instance credentials to anything that asks."""
    with pytest.raises(UrlRejected):
        resolve_and_vet("http://169.254.169.254/latest/meta-data/iam/", CONFIG)


def test_the_apps_own_address_is_blocked():
    """Otherwise a visitor can aim the fetcher at the service itself."""
    with pytest.raises(UrlRejected) as exc:
        resolve_and_vet("http://203.0.113.10/", CONFIG)
    assert exc.value.reason == "own_address"


def test_public_looking_name_resolving_to_loopback_is_blocked(stub_dns):
    """The case that proves the *resolved address* is checked, not the string.

    localtest.me is an ordinary public DNS record whose A record is 127.0.0.1. A guard
    that inspects the hostname passes it; a guard that resolves first does not.
    """
    stub_dns({"localtest.me": "127.0.0.1"})
    with pytest.raises(UrlRejected) as exc:
        resolve_and_vet("http://localtest.me/", CONFIG)
    assert exc.value.reason == "loopback_address"


def test_name_resolving_into_private_space_is_blocked(stub_dns):
    stub_dns({"internal.example.com": "10.1.2.3"})
    with pytest.raises(UrlRejected) as exc:
        resolve_and_vet("http://internal.example.com/", CONFIG)
    assert exc.value.reason == "private_address"


def test_public_name_is_allowed_and_pinned(stub_dns):
    stub_dns({"example.com": "93.184.216.34"})
    target = resolve_and_vet("http://example.com/a", CONFIG)
    assert target.host == "example.com"
    assert target.ip == "93.184.216.34"
    assert target.port == 80


def test_unresolvable_name_reports_dns_failure(stub_dns):
    stub_dns({})
    with pytest.raises(UrlRejected) as exc:
        resolve_and_vet("http://nx.example.com/", CONFIG)
    assert exc.value.reason == "dns_resolution_failed"


def test_decimal_and_octal_encoded_loopback_are_blocked(stub_dns):
    """http://2130706433/ and http://0177.0.0.1/ are both 127.0.0.1 in disguise.

    Neither parses as a literal IP, so both go through DNS -- where a resolver that
    understands the encoding hands back loopback and the address check catches it.
    """
    stub_dns({"2130706433": "127.0.0.1", "0177.0.0.1": "127.0.0.1"})
    for url in ("http://2130706433/", "http://0177.0.0.1/"):
        with pytest.raises(UrlRejected) as exc:
            resolve_and_vet(url, CONFIG)
        assert exc.value.reason == "loopback_address"


# --- client-level behaviour --------------------------------------------------


def test_kill_switch_returns_a_state_rather_than_raising():
    """FETCH_ENABLED=false must degrade the service, not break it."""
    result = client.get("https://example.com/", FetchConfig(enabled=False))
    assert result.outcome is FetchOutcome.DISABLED
    assert result.state == "unreachable"
    assert result.html == ""


def test_blocked_url_produces_a_typed_result_not_an_exception():
    result = client.get("http://127.0.0.1/", CONFIG)
    assert result.outcome is FetchOutcome.BLOCKED_BY_POLICY
    assert result.reason == "loopback_address"


def test_redirect_into_loopback_is_blocked_on_the_second_hop(monkeypatch, stub_dns):
    """Per-hop revalidation.

    A public URL that 302s to 127.0.0.1 is the standard bypass for a guard that only
    checks the URL it was handed.
    """
    stub_dns({"example.com": "93.184.216.34"})

    hops = {"n": 0}

    class FakeResponse:
        def __init__(self, status, headers, body=b""):
            self.status_code = status
            self.headers = headers
            self.encoding = "utf-8"
            self.apparent_encoding = "utf-8"
            self._body = body

        @property
        def is_redirect(self):
            return self.status_code in (301, 302, 303, 307, 308)

        def iter_content(self, chunk_size=16384, decode_unicode=False):
            yield self._body

        def close(self):
            pass

    class FakeSession:
        trust_env = True
        max_redirects = 0
        headers: dict = {}
        proxies: dict = {}

        def mount(self, *a, **k):
            pass

        def get(self, url, **kwargs):
            hops["n"] += 1
            return FakeResponse(302, {"Location": "http://127.0.0.1/admin"})

        def close(self):
            pass

    monkeypatch.setattr(client, "_build_session", lambda *_: FakeSession())

    result = client.get("http://example.com/", CONFIG)
    assert result.outcome is FetchOutcome.BLOCKED_BY_POLICY
    assert result.reason == "loopback_address"
    assert hops["n"] == 1, "must not have issued a request to the loopback target"


def test_oversize_body_is_capped_on_decompressed_bytes(monkeypatch, stub_dns):
    """A compression bomb declares a small Content-Length and expands to gigabytes, so the
    cap is applied to what is read, not to what was advertised."""
    stub_dns({"example.com": "93.184.216.34"})

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "text/html", "Content-Length": "512"}
        encoding = "utf-8"
        apparent_encoding = "utf-8"
        is_redirect = False

        def iter_content(self, chunk_size=16384, decode_unicode=False):
            for _ in range(200):
                yield b"A" * 16384

        def close(self):
            pass

    class FakeSession:
        trust_env = True
        max_redirects = 0
        headers: dict = {}
        proxies: dict = {}

        def mount(self, *a, **k):
            pass

        def get(self, url, **kwargs):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(client, "_build_session", lambda *_: FakeSession())

    config = FetchConfig(max_bytes=64 * 1024)
    result = client.get("http://example.com/", config)
    assert result.outcome is FetchOutcome.TOO_LARGE


def test_non_html_content_type_is_refused(monkeypatch, stub_dns):
    stub_dns({"example.com": "93.184.216.34"})

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}
        encoding = "utf-8"
        apparent_encoding = "utf-8"
        is_redirect = False

        def iter_content(self, chunk_size=16384, decode_unicode=False):
            yield b"%PDF-1.4"

        def close(self):
            pass

    class FakeSession:
        trust_env = True
        max_redirects = 0
        headers: dict = {}
        proxies: dict = {}

        def mount(self, *a, **k):
            pass

        def get(self, url, **kwargs):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(client, "_build_session", lambda *_: FakeSession())
    result = client.get("http://example.com/", CONFIG)
    assert result.outcome is FetchOutcome.UNSUPPORTED_TYPE


def test_challenge_page_is_not_reported_as_a_successful_scrape():
    """A bot wall returns HTTP 200 with HTML describing the challenge, not the site.
    Counting it as scraped would feed the model Cloudflare's markup as evidence."""
    assert safety.looks_like_challenge("<title>Just a moment...</title>", None)
    assert safety.looks_like_challenge("<p>Checking your browser before accessing</p>", None)
    assert not safety.looks_like_challenge("<title>Acme Bank</title><p>Welcome</p>", None)


def test_fetch_states_are_three_not_two():
    from phishguard.fetch.client import FetchResult

    assert FetchResult(FetchOutcome.OK, "u").state == "scraped"
    assert FetchResult(FetchOutcome.CHALLENGE_DETECTED, "u").state == "challenged"
    assert FetchResult(FetchOutcome.TIMEOUT, "u").state == "unreachable"
    assert FetchResult(FetchOutcome.DNS_FAIL, "u").state == "unreachable"


def test_robots_blanket_disallow_parsing():
    from phishguard.fetch.client import _has_blanket_disallow

    assert _has_blanket_disallow("User-agent: *\nDisallow: /")
    assert not _has_blanket_disallow("User-agent: *\nDisallow: /admin")
    assert not _has_blanket_disallow("User-agent: badbot\nDisallow: /")
    assert not _has_blanket_disallow("")
    # A disallow under a different agent must not be attributed to the wildcard group.
    assert not _has_blanket_disallow("User-agent: *\nDisallow: /a\n\nUser-agent: x\nDisallow: /")
