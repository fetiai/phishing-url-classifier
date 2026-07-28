"""The 25 page-derived features.

THESE RULES ARE HYPOTHESES
==========================

This is the most important caveat in the codebase. The source dataset's own extraction
code is not available, and neither is the crawled HTML corpus its rows were computed from.
Every rule below is a reconstruction inferred from a feature's name and a plausible
implementation. They may not match how the training data was actually produced, and no
amount of reading them will establish whether they do.

The response is measurement, not assertion. The agreement harness compares live extraction
against the dataset's own values for the same URLs, gates each feature, and *demotes* the
ones that fail -- a demoted extractor returns None permanently, its value is always
imputed, and the interface labels it as not reliably extractable. Nothing here is trusted
because it looks reasonable.

Each extractor returns None when it cannot compute a value, rather than a sentinel such as
0 or -1. A sentinel is indistinguishable from a real measurement once it reaches the
imputer, so it would quietly become evidence.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from phishguard import constants
from phishguard.config import APP, RefScope
from phishguard.features.domainutil import registrable_domain, site_key

META_REFRESH_URL = re.compile(r"url\s*=\s*['\"]?([^'\";]+)", re.IGNORECASE)


def parse_html(raw_html: str) -> BeautifulSoup:
    """Parse with lxml, for speed and for tolerance of the malformed markup that
    phishing pages routinely serve."""
    return BeautifulSoup(raw_html, "lxml")


def extract_title(soup: BeautifulSoup) -> str | None:
    if soup.title is None:
        return None
    text = soup.title.get_text(strip=True)
    return text or None


def _resolve(final_url: str, ref: str) -> str:
    return urljoin(final_url, ref)


def _attr_str(tag: Any, name: str) -> str:
    value = tag.get(name)
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _rel_tokens(tag: Any) -> set[str]:
    rel = tag.get("rel")
    if rel is None:
        return set()
    if isinstance(rel, list):
        return {str(r).lower() for r in rel}
    return {part.lower() for part in str(rel).split()}


def _meta_content(soup: BeautifulSoup, *, name: str = "", prop: str = "") -> str | None:
    selector: dict[str, Any] = {"name": name} if name else {"property": prop}
    tag = soup.find("meta", attrs=selector)
    if tag is None:
        return None
    content = _attr_str(tag, "content").strip()
    return content or None


def _references(soup: BeautifulSoup, scope: RefScope) -> list[tuple[str, str]]:
    """(tag_name, reference) pairs in scope for the three reference-count features.

    Which scope is correct is not knowable by inspection -- these three features are the
    least certain of the 25 -- so both are implemented and the choice is settled by
    measured agreement rather than by argument.
    """
    if scope == "anchor":
        return [("a", _attr_str(a, "href")) for a in soup.find_all("a")]

    pairs: list[tuple[str, str]] = []
    for tag_name, attr in constants.RESOURCE_REF_TAGS:
        for tag in soup.find_all(tag_name):
            pairs.append((tag_name, _attr_str(tag, attr)))
    return pairs


def _is_empty_ref(ref: str) -> bool:
    return ref.strip().lower() in constants.EMPTY_REFS


def extract_html_features(
    raw_html: str,
    soup: BeautifulSoup,
    final_url: str,
    redirect_chain: list[str],
    *,
    ref_scope: RefScope | None = None,
    robots_allowed: bool | None = None,
    demoted: frozenset[str] = frozenset(),
) -> dict[str, float | int | None]:
    """Compute the 25 page features plus the intermediate Title.

    ``demoted`` names features the agreement gate rejected. They return None regardless of
    what the page contains, so a feature we cannot extract reliably never masquerades as
    evidence.
    """
    scope: RefScope = ref_scope or APP.ref_scope

    # The raw decoded response text, before any prettifying. LineOfCode and
    # LargestLineLength are properties of the bytes as served; running them over a
    # re-serialised tree would measure BeautifulSoup's formatter instead of the page.
    lines = raw_html.splitlines()

    title = extract_title(soup)
    text = soup.get_text(" ", strip=True).lower()
    page_rd = site_key(final_url)
    haystack = f"{text} {title or ''} {final_url}".lower()

    features: dict[str, float | int | None] = {}

    features["Title"] = title  # type: ignore[assignment]
    features["HasTitle"] = 1 if title else 0

    features["LineOfCode"] = len(lines)
    features["LargestLineLength"] = max((len(line) for line in lines), default=0)

    # No second request for /favicon.ico: the fetch budget is one page request, plus
    # optionally robots.txt.
    features["HasFavicon"] = 1 if any(
        _rel_tokens(link) & constants.FAVICON_REL_TOKENS and _attr_str(link, "href").strip()
        for link in soup.find_all("link")
    ) else 0

    if robots_allowed is not None:
        features["Robots"] = 1 if robots_allowed else 0
    else:
        meta_robots = (_meta_content(soup, name="robots") or "").lower()
        features["Robots"] = 0 if ("noindex" in meta_robots or "none" in meta_robots) else 1

    viewport = _meta_content(soup, name="viewport") or ""
    has_media_query = any(
        "@media" in (style.get_text() or "") for style in soup.find_all("style")
    )
    features["IsResponsive"] = 1 if ("width=" in viewport.lower() or has_media_query) else 0

    meta_refresh_targets = []
    for meta in soup.find_all("meta"):
        if _attr_str(meta, "http-equiv").strip().lower() != "refresh":
            continue
        match = META_REFRESH_URL.search(_attr_str(meta, "content"))
        if match:
            meta_refresh_targets.append(_resolve(final_url, match.group(1).strip()))

    # HTTP hops actually followed, plus meta refreshes carrying a URL. JavaScript
    # redirects are excluded because no JavaScript is executed -- counting them would
    # require claiming knowledge the fetcher does not have.
    features["NoOfURLRedirect"] = len(redirect_chain) + len(meta_refresh_targets)
    features["NoOfSelfRedirect"] = sum(
        1
        for target in [*redirect_chain, *meta_refresh_targets]
        if page_rd is not None and site_key(target) == page_rd
    )

    features["HasDescription"] = 1 if (
        _meta_content(soup, name="description") or _meta_content(soup, prop="og:description")
    ) else 0

    popup_sources = [script.get_text() or "" for script in soup.find_all("script")]
    for tag in soup.find_all(True):
        for attr, value in tag.attrs.items():
            if attr.lower().startswith("on"):
                popup_sources.append(value if isinstance(value, str) else " ".join(value))
    popup_text = "\n".join(popup_sources)
    features["NoOfPopup"] = sum(
        len(pattern.findall(popup_text)) for pattern in constants.POPUP_PATTERNS
    )

    features["NoOfiFrame"] = len(soup.find_all(["iframe", "frame"]))

    external_form = False
    for form in soup.find_all("form"):
        action = _attr_str(form, "action").strip()
        # Empty, "#", and relative actions post back to the same site.
        if not action or action == "#":
            continue
        action_rd = site_key(_resolve(final_url, action))
        if action_rd is not None and page_rd is not None and action_rd != page_rd:
            external_form = True
            break
    features["HasExternalFormSubmit"] = 1 if external_form else 0

    features["HasSocialNet"] = 1 if any(
        registrable_domain(_resolve(final_url, _attr_str(a, "href"))) in constants.SOCIAL_DOMAINS
        for a in soup.find_all("a")
        if _attr_str(a, "href").strip()
    ) else 0

    has_submit = bool(
        soup.select('input[type="submit"]') or soup.select('button[type="submit"]')
    )
    if not has_submit:
        # A <button> with no type attribute defaults to submit inside a form.
        has_submit = any(
            button.get("type") is None and button.find_parent("form") is not None
            for button in soup.find_all("button")
        )
    features["HasSubmitButton"] = 1 if has_submit else 0

    features["HasHiddenFields"] = 1 if soup.select('input[type="hidden"]') else 0
    features["HasPasswordField"] = 1 if soup.select('input[type="password"]') else 0

    features["Bank"] = 1 if constants.BANK_PATTERN.search(haystack) else 0
    features["Pay"] = 1 if constants.PAY_PATTERN.search(haystack) else 0
    features["Crypto"] = 1 if constants.CRYPTO_PATTERN.search(haystack) else 0

    footer_text = " ".join(f.get_text(" ", strip=True) for f in soup.find_all("footer"))
    tail = text[-1500:]
    features["HasCopyrightInfo"] = 1 if (
        constants.COPYRIGHT_PATTERN.search(tail)
        or constants.COPYRIGHT_PATTERN.search(footer_text)
        or constants.COPYRIGHT_PATTERN.search(raw_html[-2000:])
    ) else 0

    features["NoOfImage"] = len(soup.find_all("img"))
    features["NoOfCSS"] = len(
        [link for link in soup.find_all("link") if "stylesheet" in _rel_tokens(link)]
    ) + len(soup.find_all("style"))
    features["NoOfJS"] = len(soup.find_all("script"))

    self_refs = 0
    empty_refs = 0
    external_refs = 0
    for _tag_name, ref in _references(soup, scope):
        if _is_empty_ref(ref):
            empty_refs += 1
            continue
        target_rd = site_key(_resolve(final_url, ref))
        if target_rd is None:
            # Unresolvable to a host: mailto:, tel:, data: and similar. Not a reference to
            # anywhere, so it is not counted as self or external.
            continue
        if page_rd is not None and target_rd == page_rd:
            self_refs += 1
        else:
            external_refs += 1

    features["NoOfSelfRef"] = self_refs
    features["NoOfEmptyRef"] = empty_refs
    features["NoOfExternalRef"] = external_refs

    for name in demoted:
        if name in features:
            features[name] = None

    return features


def empty_html_features(demoted: frozenset[str] = frozenset()) -> dict[str, float | int | None]:
    """Every page feature unknown, for when there is no page.

    Used when the fetch failed or was disabled. All None rather than zeros: a zero
    iframe count is a measurement, an absent one is not, and the imputer must be able to
    tell them apart.
    """
    from phishguard import schema

    unknown: dict[str, float | int | None] = dict.fromkeys(schema.HTML_FEATURES)
    unknown["Title"] = None
    unknown["HasTitle"] = None
    for name in demoted:
        unknown[name] = None
    return unknown
