"""Registrable-domain helper.

Registrable domain rather than netloc, so ``login.example.co.uk`` and
``www.example.co.uk`` are recognised as the same site. Comparing netlocs would count every
subdomain of a page as external, which would make the self/external reference counts
measure subdomain structure instead of what they are meant to.

The public-suffix list is loaded from a snapshot rather than fetched. A container that
phones home on first use has an outbound dependency in its startup path, and -- worse --
the feature values it produces would depend on when it happened to refresh.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit

import tldextract

#: suffix_list_urls=None disables network refresh entirely; the bundled snapshot is used.
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)


@lru_cache(maxsize=8192)
def registrable_domain(url_or_host: str | None) -> str | None:
    """The registrable domain of a URL or bare hostname, lowercased.

    Returns None when there is nothing resolvable to a registrable name -- a relative
    reference, an empty string, a bare IP with no suffix, or a scheme like ``mailto:``.
    """
    if not url_or_host:
        return None

    candidate = url_or_host.strip()
    if not candidate:
        return None

    if "//" in candidate:
        host = urlsplit(candidate).hostname or ""
    elif "/" in candidate or ":" in candidate:
        parsed = urlsplit(candidate if "//" in candidate else f"//{candidate}")
        host = parsed.hostname or ""
    else:
        host = candidate

    if not host:
        return None

    extracted = _EXTRACTOR(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()

    # An IP literal has no registrable domain, but two references to the same literal are
    # still the same host, so returning it keeps self/external comparison meaningful.
    if extracted.ipv4 or extracted.ipv6:
        return host.lower()

    return None


@lru_cache(maxsize=8192)
def site_key(url_or_host: str | None) -> str | None:
    """An identity for "which site is this", for comparing two references.

    The registrable domain where the suffix is recognised, and the bare hostname where it
    is not. The fallback matters: the public-suffix snapshot does not know every TLD --
    newly delegated ones, internal names, and reserved suffixes such as ``.test`` are all
    absent -- and returning None for those would drop the reference from the self and
    external counts entirely. A link to an unrecognised host is still a link somewhere,
    and silently discarding it would understate exactly the unusual domains a phishing
    page is most likely to use.
    """
    registrable = registrable_domain(url_or_host)
    if registrable is not None:
        return registrable

    if not url_or_host:
        return None

    candidate = url_or_host.strip()
    if not candidate:
        return None

    parts = urlsplit(candidate)

    # A scheme with no authority -- mailto:, tel:, data:, javascript: -- names no host.
    # This is decided before anything else, because "data:text/plain,x" contains a slash
    # and would otherwise be parsed as though it had a path.
    if parts.scheme and not parts.netloc:
        return None

    host = parts.hostname if parts.netloc else urlsplit(f"//{candidate}").hostname
    return host.lower() if host else None


def same_site(a: str | None, b: str | None) -> bool:
    """True when both refer to the same site."""
    key_a = site_key(a)
    key_b = site_key(b)
    return key_a is not None and key_a == key_b
