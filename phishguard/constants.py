"""Frozen keyword and domain lists used by the page-feature extractors.

These are versioned. Changing a list changes the feature distribution and therefore
invalidates the trained models, so ``KEYWORD_LIST_VERSION`` is recorded in the artifact
manifest and a bundle whose version does not match the running code is refused at load.

Matching is whole-word throughout, so "pay" does not fire on "display" and multi-word
entries such as "seed phrase" match as phrases rather than as their parts.
"""

from __future__ import annotations

import re
from typing import Final

KEYWORD_LIST_VERSION: Final[str] = "1"

SOCIAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "pinterest.com",
        "tiktok.com",
        "t.me",
        "telegram.org",
        "whatsapp.com",
        "reddit.com",
        "github.com",
        "vk.com",
        "weibo.com",
    }
)

BANK_KEYWORDS: Final[tuple[str, ...]] = (
    "bank",
    "banking",
    "banco",
    "netbanking",
    "account",
    "savings",
    "checking",
    "ifsc",
    "iban",
    "swift",
    "credit union",
    "debit card",
)

PAY_KEYWORDS: Final[tuple[str, ...]] = (
    "pay",
    "payment",
    "paypal",
    "checkout",
    "invoice",
    "billing",
    "visa",
    "mastercard",
    "amex",
    "stripe",
    "payoneer",
    "western union",
    "transfer",
    "remit",
)

CRYPTO_KEYWORDS: Final[tuple[str, ...]] = (
    "crypto",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "wallet",
    "blockchain",
    "binance",
    "coinbase",
    "metamask",
    "ledger",
    "trezor",
    "seed phrase",
    "usdt",
    "nft",
)

#: Anchor targets that go nowhere. Compared case-insensitively after stripping whitespace.
EMPTY_REFS: Final[frozenset[str]] = frozenset(
    {
        "",
        "#",
        "javascript:void(0)",
        "javascript:void(0);",
        "javascript:;",
        "about:blank",
    }
)

#: Elements carrying a reference, used when the reference-count features are configured to
#: count all resources rather than anchors alone.
RESOURCE_REF_TAGS: Final[tuple[tuple[str, str], ...]] = (
    ("a", "href"),
    ("link", "href"),
    ("script", "src"),
    ("img", "src"),
    ("iframe", "src"),
    ("form", "action"),
    ("source", "src"),
)

FAVICON_REL_TOKENS: Final[frozenset[str]] = frozenset(
    {"icon", "shortcut", "apple-touch-icon", "mask-icon"}
)

POPUP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"window\.open\s*\(", re.IGNORECASE),
    re.compile(r"\balert\s*\(", re.IGNORECASE),
    re.compile(r"\bconfirm\s*\(", re.IGNORECASE),
    re.compile(r"\bprompt\s*\(", re.IGNORECASE),
)

COPYRIGHT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"©|&copy;|\bcopyright\b|all rights reserved", re.IGNORECASE
)

#: Signatures of a bot wall or interstitial returning HTTP 200. A page matching these
#: describes the challenge rather than the site, so treating it as a successful scrape
#: would feed the model features of Cloudflare's markup instead of the target's.
CHALLENGE_MARKERS: Final[tuple[str, ...]] = (
    "just a moment",
    "checking your browser",
    "enable javascript and cookies to continue",
    "cf-browser-verification",
    "cf_chl_opt",
    "attention required! | cloudflare",
    "ddos protection by",
    "please verify you are a human",
    "captcha-delivery.com",
    "px-captcha",
    "incapsula incident id",
    "access denied",
)


def _compile_keywords(keywords: tuple[str, ...]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


BANK_PATTERN: Final[re.Pattern[str]] = _compile_keywords(BANK_KEYWORDS)
PAY_PATTERN: Final[re.Pattern[str]] = _compile_keywords(PAY_KEYWORDS)
CRYPTO_PATTERN: Final[re.Pattern[str]] = _compile_keywords(CRYPTO_KEYWORDS)
