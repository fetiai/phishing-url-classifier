"""The 25 page-derived extractors, against hand-written markup.

These tests establish that each extractor *runs* and implements the rule it claims. They
say nothing about whether the rule matches how the training data was produced -- that is
unknowable by inspection, because the dataset's own extraction code is not available, and
it is what the agreement gate measures instead.

So a green run here means "the extractor does what it says", not "the extractor is right".
The distinction is the whole reason the demotion mechanism exists.
"""

from __future__ import annotations

import pytest

from phishguard import schema
from phishguard.features.domainutil import registrable_domain, same_site, site_key
from phishguard.features.html_features import (
    empty_html_features,
    extract_html_features,
    extract_title,
    parse_html,
)

PAGE_URL = "https://acme.example.com/login"


def extract(html: str, *, url: str = PAGE_URL, chain: list[str] | None = None, **kwargs):
    soup = parse_html(html)
    return extract_html_features(html, soup, url, chain or [], **kwargs)


# --- coverage ----------------------------------------------------------------


def test_every_page_feature_is_produced():
    features = extract("<html><body></body></html>")
    missing = set(schema.HTML_FEATURES) - set(features)
    assert not missing, f"extractor produces no value for: {sorted(missing)}"


def test_malformed_markup_does_not_raise():
    """Phishing pages routinely serve broken HTML; refusing to parse it would mean
    refusing to classify exactly the pages that matter most."""
    for broken in (
        "<html><body><a href=",
        "<<<>>>",
        "<html><body><div><p>unclosed",
        "",
        "not html at all",
        "<html><body>" + "<div>" * 500,
    ):
        features = extract(broken)
        assert set(schema.HTML_FEATURES) <= set(features)


def test_unknown_page_is_all_none_not_zero():
    """A zero iframe count is a measurement; an absent one is not. Collapsing the two
    would let 'we could not look' become evidence of 'there was nothing there'."""
    unknown = empty_html_features()
    assert set(schema.HTML_FEATURES) <= set(unknown)
    assert all(unknown[name] is None for name in schema.HTML_FEATURES)


# --- individual rules --------------------------------------------------------


def test_title():
    assert extract_title(parse_html("<title>  Hello  </title>")) == "Hello"
    assert extract_title(parse_html("<title></title>")) is None
    assert extract_title(parse_html("<html></html>")) is None


def test_has_title_reflects_a_non_empty_title():
    assert extract("<title>x</title>")["HasTitle"] == 1
    assert extract("<title>  </title>")["HasTitle"] == 0
    assert extract("<html></html>")["HasTitle"] == 0


def test_line_metrics_measure_the_raw_bytes_not_the_reparsed_tree():
    html = "<html>\n<body>\n<p>" + "x" * 300 + "</p>\n</body>\n</html>"
    features = extract(html)
    assert features["LineOfCode"] == 5
    assert features["LargestLineLength"] == 307


def test_favicon_requires_a_href():
    assert extract('<link rel="icon" href="/f.ico">')["HasFavicon"] == 1
    assert extract('<link rel="shortcut icon" href="/f.ico">')["HasFavicon"] == 1
    assert extract('<link rel="icon" href="">')["HasFavicon"] == 0
    assert extract('<link rel="stylesheet" href="/a.css">')["HasFavicon"] == 0


def test_robots_prefers_the_fetched_file_over_the_meta_proxy():
    """The fetched file is authoritative when available; the meta tag is a fallback."""
    noindex = '<meta name="robots" content="noindex">'
    assert extract(noindex)["Robots"] == 0
    assert extract(noindex, robots_allowed=True)["Robots"] == 1
    assert extract("<html></html>", robots_allowed=False)["Robots"] == 0
    assert extract("<html></html>")["Robots"] == 1


def test_responsive_from_viewport_or_media_query():
    assert extract('<meta name="viewport" content="width=device-width">')["IsResponsive"] == 1
    assert extract("<style>@media (max-width:600px){a{}}</style>")["IsResponsive"] == 1
    assert extract('<meta name="viewport" content="initial-scale=1">')["IsResponsive"] == 0
    assert extract("<html></html>")["IsResponsive"] == 0


def test_redirect_counts_combine_http_hops_and_meta_refresh():
    html = '<meta http-equiv="refresh" content="5; url=https://acme.example.com/next">'
    features = extract(html, chain=["https://acme.example.com/a", "https://other.test/b"])
    assert features["NoOfURLRedirect"] == 3
    # Two of the three point back at the page's own registrable domain.
    assert features["NoOfSelfRedirect"] == 2


def test_javascript_redirects_are_not_counted():
    """No JavaScript is executed, so claiming to have detected a JS redirect would be
    asserting knowledge the fetcher does not have."""
    features = extract('<script>window.location="https://evil.test/"</script>')
    assert features["NoOfURLRedirect"] == 0


def test_description_from_either_meta_or_open_graph():
    assert extract('<meta name="description" content="x">')["HasDescription"] == 1
    assert extract('<meta property="og:description" content="x">')["HasDescription"] == 1
    assert extract('<meta name="description" content="">')["HasDescription"] == 0


def test_popup_count_spans_scripts_and_event_handlers():
    html = """<script>window.open('a'); alert('b');</script>
              <div onclick="confirm('c')" onmouseover="prompt('d')"></div>"""
    assert extract(html)["NoOfPopup"] == 4


def test_iframe_count_includes_legacy_frame():
    assert extract("<iframe src='a'></iframe><frame src='b'>")["NoOfiFrame"] == 2


def test_external_form_submit():
    assert extract('<form action="https://evil.test/x"></form>')["HasExternalFormSubmit"] == 1
    # Empty, "#", and relative actions all post back to the same site.
    assert extract('<form action=""></form>')["HasExternalFormSubmit"] == 0
    assert extract('<form action="#"></form>')["HasExternalFormSubmit"] == 0
    assert extract('<form action="/submit"></form>')["HasExternalFormSubmit"] == 0
    # A different subdomain is the same registrable domain, so it is internal.
    assert extract('<form action="https://www.acme.example.com/s"></form>')[
        "HasExternalFormSubmit"
    ] == 0


def test_social_net_detection():
    assert extract('<a href="https://www.facebook.com/acme">f</a>')["HasSocialNet"] == 1
    assert extract('<a href="https://t.me/acme">t</a>')["HasSocialNet"] == 1
    assert extract('<a href="https://example.org/">x</a>')["HasSocialNet"] == 0


def test_submit_button_includes_a_typeless_button_inside_a_form():
    """Inside a form, a <button> with no type attribute defaults to submit."""
    assert extract('<form><button>Go</button></form>')["HasSubmitButton"] == 1
    assert extract('<input type="submit">')["HasSubmitButton"] == 1
    assert extract('<button type="submit">Go</button>')["HasSubmitButton"] == 1
    # Outside a form, and with an explicit non-submit type, it is not a submit control.
    assert extract("<button>Go</button>")["HasSubmitButton"] == 0
    assert extract('<form><button type="button">Go</button></form>')["HasSubmitButton"] == 0


def test_hidden_and_password_fields():
    assert extract('<input type="hidden">')["HasHiddenFields"] == 1
    assert extract('<input type="password">')["HasPasswordField"] == 1
    assert extract('<input type="text">')["HasPasswordField"] == 0


@pytest.mark.parametrize(
    "text,bank,pay,crypto",
    [
        ("Welcome to your bank account", 1, 0, 0),
        ("Complete your payment via paypal", 0, 1, 0),
        ("Connect your metamask wallet", 0, 0, 1),
        ("Enter your seed phrase", 0, 0, 1),
        ("Nothing relevant here", 0, 0, 0),
    ],
)
def test_keyword_features(text, bank, pay, crypto):
    features = extract(f"<body><p>{text}</p></body>", url="https://x.example/")
    assert (features["Bank"], features["Pay"], features["Crypto"]) == (bank, pay, crypto)


def test_keyword_matching_is_whole_word():
    """Substring matching would fire 'pay' on 'display' and make the feature noise."""
    features = extract("<body><p>display the repayment</p></body>", url="https://x.example/")
    assert features["Pay"] == 0


def test_copyright_detection():
    assert extract("<footer>© 2024 Acme</footer>")["HasCopyrightInfo"] == 1
    assert extract("<body>All rights reserved</body>")["HasCopyrightInfo"] == 1
    assert extract("<body>Copyright Acme</body>")["HasCopyrightInfo"] == 1
    assert extract("<body>nothing</body>")["HasCopyrightInfo"] == 0


def test_resource_counts():
    html = """<img src="1"><img src="2">
              <link rel="stylesheet" href="a.css"><style>x{}</style>
              <script src="a.js"></script><script>var x=1</script>"""
    features = extract(html)
    assert features["NoOfImage"] == 2
    assert features["NoOfCSS"] == 2
    assert features["NoOfJS"] == 2


# --- the reference-count features and their scope ----------------------------

REF_HTML = """
<a href="/about">self relative</a>
<a href="https://acme.example.com/x">self absolute</a>
<a href="https://www.acme.example.com/y">self subdomain</a>
<a href="https://other.test/z">external</a>
<a href="#">empty hash</a>
<a href="javascript:void(0)">empty js</a>
<a>no href at all</a>
<a href="mailto:a@b.c">mail</a>
<img src="https://cdn.test/i.png">
<script src="https://cdn.test/s.js"></script>
"""


def test_reference_counts_in_anchor_scope():
    features = extract(REF_HTML, ref_scope="anchor")
    assert features["NoOfSelfRef"] == 3
    assert features["NoOfExternalRef"] == 1
    # "#", javascript:void(0), and the href-less anchor.
    assert features["NoOfEmptyRef"] == 3


def test_reference_counts_in_all_resources_scope():
    """Widening the scope must pull in the img and script sources, and nothing else."""
    anchors = extract(REF_HTML, ref_scope="anchor")
    resources = extract(REF_HTML, ref_scope="all_resources")
    assert resources["NoOfExternalRef"] == anchors["NoOfExternalRef"] + 2
    assert resources["NoOfSelfRef"] == anchors["NoOfSelfRef"]


def test_scope_is_configurable_without_a_code_change():
    """Which definition is correct is settled by measured agreement, not by argument, so
    both must be reachable from configuration alone."""
    assert extract(REF_HTML, ref_scope="anchor") != extract(REF_HTML, ref_scope="all_resources")


def test_unresolvable_references_are_neither_self_nor_external():
    features = extract('<a href="mailto:x@y.z">m</a><a href="tel:+1">t</a>', ref_scope="anchor")
    assert features["NoOfSelfRef"] == 0
    assert features["NoOfExternalRef"] == 0
    assert features["NoOfEmptyRef"] == 0


# --- demotion ----------------------------------------------------------------


def test_demoted_features_return_none_regardless_of_the_page():
    """A feature that failed its agreement gate must never produce a value again, however
    confidently the page would have supplied one."""
    demoted = frozenset({"NoOfSelfRef", "NoOfExternalRef"})
    features = extract(REF_HTML, demoted=demoted)
    assert features["NoOfSelfRef"] is None
    assert features["NoOfExternalRef"] is None
    assert features["NoOfEmptyRef"] is not None


# --- registrable domain ------------------------------------------------------


def test_registrable_domain_handles_multi_part_suffixes():
    assert registrable_domain("https://login.example.co.uk/a") == "example.co.uk"
    assert registrable_domain("https://www.example.com/") == "example.com"
    assert registrable_domain("example.com") == "example.com"
    assert registrable_domain("") is None
    assert registrable_domain(None) is None


def test_same_site_ignores_subdomains():
    assert same_site("https://www.example.co.uk/a", "https://login.example.co.uk/b")
    assert not same_site("https://example.com", "https://example.org")


def test_unrecognised_suffix_still_identifies_a_site():
    """The suffix snapshot does not know every TLD -- newly delegated ones, internal
    names, and reserved suffixes such as .test are all absent. Returning nothing for those
    would drop the reference from both counts, understating exactly the unusual domains a
    phishing page is most likely to link to."""
    assert registrable_domain("https://other.test/z") is None
    assert site_key("https://other.test/z") == "other.test"
    assert same_site("https://other.test/a", "https://other.test/b")
    assert not same_site("https://other.test/a", "https://elsewhere.test/b")


def test_references_to_unrecognised_suffixes_are_counted_as_external():
    features = extract('<a href="https://weird.test/x">w</a>', ref_scope="anchor")
    assert features["NoOfExternalRef"] == 1
    assert features["NoOfSelfRef"] == 0


def test_schemes_without_a_host_have_no_site_key():
    for ref in ("mailto:a@b.c", "tel:+15551234", "data:text/plain,x"):
        assert site_key(ref) is None
