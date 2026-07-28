"""URL-derived features, ported from the original notebook.

PORT CONTRACT
=============

The 26 ``fill_*`` functions below are ported **verbatim**. Every regex, every threshold,
every edge case and every quirk is preserved exactly. Only three changes are permitted:

  1. Dropping the trailing demonstration statements each notebook cell appended.
  2. Dropping ``print()`` calls.
  3. Threading fitted statistics in as explicit keyword parameters instead of computing
     them inline from the input frame.

No improvements. No bug fixes. No renaming. Where a function looks wrong it is still
ported as-is and the observation is recorded below rather than repaired.

The severity is deliberate: these functions define what the training data *means*.
Changing one silently changes the feature distribution the models were fitted on, and no
metric would reveal it -- the numbers would simply be describing something else. The
fidelity tests run these implementations and a frozen copy of the originals side by side
over a 2,000-row sample and assert exact frame equality, with no tolerance.

Change 3 is the one that matters for correctness. ``fill_tld_legitimate_prob`` originally
computed its own skew, median/mean and per-TLD group means from whatever frame it was
handed. At training time that frame is 112,323 rows; at serving time it would be a single
row, so the same URL would get a different feature value depending on what it was batched
with. Passing the statistics in makes the function a pure row-wise mapping, which is what
makes single-row inference equal batch inference.

PRESERVED DEFECTS
=================

``detect_advanced_obfuscation`` rule 4 tests whether the *reversed* URL matches
``^[a-zA-Z0-9.\\-]+$``. A string matches that pattern exactly when its reversal does, so
the rule fires for any URL consisting solely of letters, digits, dots and hyphens --
including every bare domain with no scheme. This is almost certainly not what "reversed
strings" was meant to detect. Preserved.

``fill_has_obfuscation`` ends with ``.apply(lambda x: 1 if x == True else 0)``, which
rewrites the whole column rather than only the filled cells. Preserved.

The skew threshold here is ``abs(skew) > 1``; the numeric imputer uses ``abs(skew) > 3``.
The two are genuinely different and both are preserved.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


def scalar_float(value: Any) -> float:
    """Coerce a pandas reduction result to a plain float.

    Reductions such as ``.skew()`` and ``.median()`` are typed as a wide union covering
    every dtype a Series might hold. On the numeric columns here the value is always a
    float; this narrows it so arithmetic and comparisons type-check, and it is an identity
    on the values that actually occur.
    """
    return float(value)


# ---------------------------------------------------------------------------
# Fit-time statistic
# ---------------------------------------------------------------------------


def calculate_char_prob(df: pd.DataFrame, url_col: str) -> dict[str, float]:
    """Corpus-wide alphanumeric character frequency table.

    This is a *fit-time* statistic. It is computed once over the training split and
    thereafter passed into :func:`fill_url_char_prob` as data. It must never be recomputed
    at serving time -- a single-row corpus would give every character in that one URL a
    probability of roughly 1/len(url), which has nothing to do with the training
    distribution the models learned against.
    """
    char_count: dict[str, int] = {}
    total_chars = 0
    for url in df[url_col].dropna():
        for char in url.lower():
            if char.isalnum():  # Only consider alphanumeric characters
                char_count[char] = char_count.get(char, 0) + 1
                total_chars += 1
    return {char: count / total_chars for char, count in char_count.items()}


# ---------------------------------------------------------------------------
# The 26 fill_* functions
# ---------------------------------------------------------------------------


def fill_url_length(
    data: pd.DataFrame, url_col: str = "URL", url_length_col: str = "URLLength"
) -> pd.DataFrame:
    data[url_length_col] = data.apply(
        lambda row: len(str(row[url_col]))
        if pd.isnull(row[url_length_col]) and pd.notnull(row[url_col])
        else row[url_length_col],
        axis=1,
    )
    return data


def fill_domain(
    data: pd.DataFrame, url_col: str = "URL", domain_col: str = "Domain"
) -> pd.DataFrame:
    def get_domain(url: Any) -> Any:
        return urlparse(url).netloc if pd.notnull(url) else None

    data[domain_col] = data.apply(
        lambda row: get_domain(row[url_col])
        if pd.isnull(row[domain_col]) and pd.notnull(row[url_col])
        else row[domain_col],
        axis=1,
    )
    return data


def fill_domain_length(
    data: pd.DataFrame, domain_col: str = "Domain", domain_length_col: str = "DomainLength"
) -> pd.DataFrame:
    data[domain_length_col] = data.apply(
        lambda row: len(str(row[domain_col]))
        if pd.isnull(row[domain_length_col]) and pd.notnull(row[domain_col])
        else row[domain_length_col],
        axis=1,
    )
    return data


def fill_is_domain_ip(
    data: pd.DataFrame, domain_col: str = "Domain", is_domain_ip_col: str = "IsDomainIP"
) -> pd.DataFrame:
    def is_ipaddress(domain: Any) -> bool:
        if pd.notnull(domain):
            ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
            if re.match(ip_pattern, domain):
                parts = domain.split(".")
                return all(0 <= int(part) <= 255 for part in parts)
        return False

    data[is_domain_ip_col] = data.apply(
        lambda row: (1 if is_ipaddress(row[domain_col]) else 0)
        if pd.isnull(row[is_domain_ip_col]) and pd.notnull(row[domain_col])
        else row[is_domain_ip_col],
        axis=1,
    )
    return data


def fill_tld(data: pd.DataFrame, domain_col: str = "Domain", tld_col: str = "TLD") -> pd.DataFrame:
    def generate_tld(domain: Any) -> Any:
        if pd.notnull(domain):
            parts = domain.split(".")
            if len(parts) > 1:
                return parts[-1]
        return np.nan

    data[tld_col] = data.apply(
        lambda row: generate_tld(row[domain_col]) if pd.isnull(row[tld_col]) else row[tld_col],
        axis=1,
    )
    return data


def fill_char_continuation_rate(
    data: pd.DataFrame, url_col: str = "URL", char_rate_col: str = "CharContinuationRate"
) -> pd.DataFrame:
    def generate_char_continuation_rate(url: Any) -> Any:
        if pd.notnull(url):
            sequences = re.findall(r"[a-zA-Z0-9_]+", url)
            total_sequence_length = sum(len(seq) for seq in sequences)
            total_url_length = len(url)
            return total_sequence_length / total_url_length if total_url_length > 0 else np.nan
        return np.nan

    data[char_rate_col] = data.apply(
        lambda row: generate_char_continuation_rate(row[url_col])
        if pd.isnull(row[char_rate_col])
        else row[char_rate_col],
        axis=1,
    )
    return data


def fill_tld_legitimate_prob(
    data: pd.DataFrame,
    *,
    tld_prob_mean: pd.Series,
    global_fill_value: float,
    tld_col: str = "TLD",
    tld_prob_col: str = "TLDLegitimateProb",
    is_domain_ip_col: str = "IsDomainIP",
) -> pd.DataFrame:
    """Fill ``TLDLegitimateProb``.

    ``tld_prob_mean`` (per-TLD group means) and ``global_fill_value`` (the median if
    ``abs(skew) > 1``, else the mean) are computed at fit time by
    :func:`compute_tld_prob_statistics` and passed in. The original computed both inside
    this function from its argument frame; the four-branch row rule below is otherwise
    byte-identical to it.
    """

    def fill_tld_legit_prob(row: Any) -> Any:
        if pd.isnull(row[tld_prob_col]):
            if row[is_domain_ip_col] == 1:
                return 0
            if pd.notnull(row[tld_col]) and row[tld_col] in tld_prob_mean.index:
                return tld_prob_mean[row[tld_col]]
            if pd.isnull(row[tld_col]) and pd.isnull(row[is_domain_ip_col]):
                return global_fill_value
        return row[tld_prob_col]

    data[tld_prob_col] = data.apply(fill_tld_legit_prob, axis=1)
    data[tld_prob_col] = data[tld_prob_col].fillna(global_fill_value)

    return data


def compute_tld_prob_statistics(
    data: pd.DataFrame,
    tld_col: str = "TLD",
    tld_prob_col: str = "TLDLegitimateProb",
) -> tuple[pd.Series, float]:
    """Fit-time half of :func:`fill_tld_legitimate_prob`, lifted out of it verbatim.

    Note the threshold: ``abs(skew) > 1`` here, against ``abs(skew) > 3`` in the numeric
    imputer. The two are genuinely different in the original and both are preserved.
    """
    skewness = scalar_float(data[tld_prob_col].skew())
    if skewness > 1 or skewness < -1:
        global_fill_value = scalar_float(data[tld_prob_col].median())
    else:
        global_fill_value = scalar_float(data[tld_prob_col].mean())

    tld_prob_mean = data.groupby(tld_col)[tld_prob_col].mean()
    return tld_prob_mean, global_fill_value


def fill_url_char_prob(
    data: pd.DataFrame,
    url_col: str = "URL",
    char_prob_col: str = "URLCharProb",
    char_prob: dict[str, float] | None = None,
) -> pd.DataFrame:
    if char_prob is None:
        raise ValueError("Character probabilities (`char_prob`) must be provided.")

    def calculate_url_char_prob(url: Any) -> Any:
        if pd.notnull(url):
            total_prob = sum(char_prob.get(char, 0) for char in url.lower() if char.isalnum())
            n = len(url)
            return total_prob / n if n > 0 else np.nan
        return np.nan

    data[char_prob_col] = data.apply(
        lambda row: calculate_url_char_prob(row[url_col])
        if pd.isnull(row[char_prob_col])
        else row[char_prob_col],
        axis=1,
    )
    return data


def fill_tld_length(
    data: pd.DataFrame, tld_col: str = "TLD", tld_length_col: str = "TLDLength"
) -> pd.DataFrame:
    def calculate_tld_length(tld: Any) -> Any:
        if pd.notnull(tld):
            return len(str(tld))
        return np.nan

    data[tld_length_col] = data.apply(
        lambda row: calculate_tld_length(row[tld_col])
        if pd.isnull(row[tld_length_col])
        else row[tld_length_col],
        axis=1,
    )
    return data


def fill_no_of_subdomains(
    data: pd.DataFrame, domain_col: str = "Domain", subdomain_col: str = "NoOfSubDomain"
) -> pd.DataFrame:
    def calculate_no_of_subdomains(domain: Any) -> Any:
        if pd.notnull(domain):
            parts = domain.split(".")
            return len(parts) - 2 if len(parts) > 2 else 0
        return np.nan

    data[subdomain_col] = data.apply(
        lambda row: calculate_no_of_subdomains(row[domain_col])
        if pd.isnull(row[subdomain_col])
        else row[subdomain_col],
        axis=1,
    )
    return data


def fill_has_obfuscation(
    data: pd.DataFrame, url_col: str = "URL", obfuscation_col: str = "HasObfuscation"
) -> pd.DataFrame:
    data[obfuscation_col] = data.apply(
        lambda row: detect_advanced_obfuscation(row[url_col])
        if pd.isnull(row[obfuscation_col]) and pd.notnull(row[url_col])
        else row[obfuscation_col],
        axis=1,
    )
    data[obfuscation_col] = data[obfuscation_col].apply(lambda x: 1 if x == True else 0)  # noqa: E712
    return data


def detect_advanced_obfuscation(url: Any) -> int:
    """The five obfuscation rules, first match wins.

    Rule 4 is preserved despite being near-certainly wrong: a string matches
    ``^[a-zA-Z0-9.\\-]+$`` exactly when its reversal does, so reversing accomplishes
    nothing and the rule fires for any scheme-less bare domain.
    """
    if pd.notnull(url):
        if len(re.findall(r"[-_]", url)) > 3:  # Rule 1: Too many special characters
            return 1
        if re.search(r"[a-zA-Z]+\d+|\d+[a-zA-Z]+", url):  # Rule 2: Mixed alphanumeric patterns
            return 1
        if len(url) % 4 == 0 and re.match(r"^[A-Za-z0-9+/]*={0,2}$", url):  # Rule 3: Base64
            try:
                base64.b64decode(url, validate=True)
                return 1
            except Exception:
                pass
        reversed_url = url[::-1]  # Rule 4: Reversed strings
        if re.match(r"^[a-zA-Z0-9.\-]+$", reversed_url):
            return 1
        if not re.search(r"[a-zA-Z]{3,}", url):  # Rule 5: Randomized strings
            return 1
        return 0  # No obfuscation detected
    return 0  # Missing values treated as no obfuscation


def fill_no_of_obfuscated_characters(
    data: pd.DataFrame, url_col: str = "URL", obf_char_col: str = "NoOfObfuscatedChar"
) -> pd.DataFrame:
    def count_obfuscated_characters(url: Any) -> Any:
        if pd.notnull(url):
            hex_count = len(re.findall(r"%[0-9a-fA-F]{2}", url))
            at_count = url.count("@")
            return hex_count + at_count
        return np.nan

    data[obf_char_col] = data.apply(
        lambda row: count_obfuscated_characters(row[url_col])
        if pd.isnull(row[obf_char_col])
        else row[obf_char_col],
        axis=1,
    )
    return data


def fill_obfuscation_ratio(
    data: pd.DataFrame,
    obf_char_col: str = "NoOfObfuscatedChar",
    url_length_col: str = "URLLength",
    obf_ratio_col: str = "ObfuscationRatio",
) -> pd.DataFrame:
    def calculate_obfuscation_ratio(no_of_obfchar: Any, url_length: Any) -> Any:
        if pd.notnull(no_of_obfchar) and pd.notnull(url_length) and url_length > 0:
            return no_of_obfchar / url_length
        return np.nan

    data[obf_ratio_col] = data.apply(
        lambda row: calculate_obfuscation_ratio(row[obf_char_col], row[url_length_col])
        if pd.isnull(row[obf_ratio_col])
        else row[obf_ratio_col],
        axis=1,
    )
    return data


def fill_no_of_letters_in_url(
    data: pd.DataFrame, url_col: str = "URL", letters_col: str = "NoOfLettersInURL"
) -> pd.DataFrame:
    def calculate_no_of_letters(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c.isalpha() for c in url)
        return np.nan

    data[letters_col] = data.apply(
        lambda row: calculate_no_of_letters(row[url_col])
        if pd.isnull(row[letters_col])
        else row[letters_col],
        axis=1,
    )
    return data


def fill_letter_ratio_in_url(
    data: pd.DataFrame,
    letters_col: str = "NoOfLettersInURL",
    url_length_col: str = "URLLength",
    ratio_col: str = "LetterRatioInURL",
) -> pd.DataFrame:
    def calculate_letter_ratio(no_of_letters: Any, url_length: Any) -> Any:
        if pd.notnull(no_of_letters) and pd.notnull(url_length) and url_length > 0:
            return no_of_letters / url_length
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_letter_ratio(row[letters_col], row[url_length_col])
        if pd.isnull(row[ratio_col])
        else row[ratio_col],
        axis=1,
    )
    return data


def fill_no_of_digits_in_url(
    data: pd.DataFrame, url_col: str = "URL", digits_col: str = "NoOfDegitsInURL"
) -> pd.DataFrame:
    def calculate_no_of_digits(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c.isdigit() for c in url)
        return np.nan

    data[digits_col] = data.apply(
        lambda row: calculate_no_of_digits(row[url_col])
        if pd.isnull(row[digits_col])
        else row[digits_col],
        axis=1,
    )
    return data


def fill_digits_ratio_in_url(
    data: pd.DataFrame, url_col: str = "URL", ratio_col: str = "DegitRatioInURL"
) -> pd.DataFrame:
    def calculate_digits_ratio(url: Any) -> Any:
        if pd.notnull(url) and len(url) > 0:
            return sum(c.isdigit() for c in url) / len(url)
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_digits_ratio(row[url_col])
        if pd.isnull(row[ratio_col])
        else row[ratio_col],
        axis=1,
    )
    return data


def fill_no_of_equals_in_url(
    data: pd.DataFrame, url_col: str = "URL", equals_col: str = "NoOfEqualsInURL"
) -> pd.DataFrame:
    def calculate_no_of_equals(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c == "=" for c in url)
        return np.nan

    data[equals_col] = data.apply(
        lambda row: calculate_no_of_equals(row[url_col])
        if pd.isnull(row[equals_col])
        else row[equals_col],
        axis=1,
    )
    return data


def fill_no_of_qmark_in_url(
    data: pd.DataFrame, url_col: str = "URL", qmark_col: str = "NoOfQMarkInURL"
) -> pd.DataFrame:
    def calculate_no_of_qmark(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c == "?" for c in url)
        return np.nan

    data[qmark_col] = data.apply(
        lambda row: calculate_no_of_qmark(row[url_col])
        if pd.isnull(row[qmark_col])
        else row[qmark_col],
        axis=1,
    )
    return data


def fill_no_of_ampersand_in_url(
    data: pd.DataFrame, url_col: str = "URL", ampersand_col: str = "NoOfAmpersandInURL"
) -> pd.DataFrame:
    def calculate_no_of_ampersand(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c == "&" for c in url)
        return np.nan

    data[ampersand_col] = data.apply(
        lambda row: calculate_no_of_ampersand(row[url_col])
        if pd.isnull(row[ampersand_col])
        else row[ampersand_col],
        axis=1,
    )
    return data


def fill_no_of_special_chars_in_url(
    data: pd.DataFrame,
    url_col: str = "URL",
    special_chars_col: str = "NoOfOtherSpecialCharsInURL",
) -> pd.DataFrame:
    def calculate_no_of_specials(url: Any) -> Any:
        if pd.notnull(url):
            return sum(c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" for c in url)
        return np.nan

    data[special_chars_col] = data.apply(
        lambda row: calculate_no_of_specials(row[url_col])
        if pd.isnull(row[special_chars_col])
        else row[special_chars_col],
        axis=1,
    )
    return data


def fill_specials_ratio_in_url(
    data: pd.DataFrame, url_col: str = "URL", ratio_col: str = "SpacialCharRatioInURL"
) -> pd.DataFrame:
    def calculate_specials_ratio(url: Any) -> Any:
        if pd.notnull(url) and len(url) > 0:
            return (
                sum(c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" for c in url)
                / len(url)
            )
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_specials_ratio(row[url_col])
        if pd.isnull(row[ratio_col])
        else row[ratio_col],
        axis=1,
    )
    return data


def fill_is_https(
    data: pd.DataFrame, url_col: str = "URL", https_col: str = "IsHTTPS"
) -> pd.DataFrame:
    def calculate_ishttps(url: Any) -> Any:
        if pd.notnull(url):
            return 1 if "https://" in url else 0
        return np.nan

    data[https_col] = data.apply(
        lambda row: calculate_ishttps(row[url_col])
        if pd.isnull(row[https_col])
        else row[https_col],
        axis=1,
    )
    return data


def fill_has_title(
    data: pd.DataFrame, title_col: str = "Title", has_title_col: str = "HasTitle"
) -> pd.DataFrame:
    data[has_title_col] = data.apply(
        lambda row: row[has_title_col]
        if not pd.isnull(row[has_title_col])
        else (1 if not pd.isnull(row[title_col]) else row[has_title_col]),
        axis=1,
    )
    return data


def fill_domain_title_match_score(
    data: pd.DataFrame,
    title_col: str = "Title",
    domain_col: str = "Domain",
    score_col: str = "DomainTitleMatchScore",
) -> pd.DataFrame:
    data[score_col] = data.apply(
        lambda row: calculate_domain_title_match_score(row[title_col], row[domain_col])
        if pd.isnull(row[score_col])
        and pd.notnull(row[title_col])
        and pd.notnull(row[domain_col])
        else row[score_col],
        axis=1,
    )
    return data


def calculate_domain_title_match_score(title: Any, domain: Any) -> float:
    tSet = title.split(" ") if pd.notnull(title) else []
    txtDomain = domain.split(".")[:-1] if pd.notnull(domain) else []
    txtDomain = [i for i in txtDomain if i != "www"]
    txtDomain = ".".join(txtDomain)

    score = 0.0
    baseScore = 100 / len(txtDomain) if len(txtDomain) > 0 else 0

    for element in tSet:
        if element in txtDomain:
            n = len(element)
            score += baseScore * n
            txtDomain = txtDomain.replace(element, "")
            if score > 99.9:
                score = 100
    return score


def fill_url_title_match_score(
    data: pd.DataFrame,
    title_col: str = "Title",
    url_col: str = "URL",
    score_col: str = "URLTitleMatchScore",
) -> pd.DataFrame:
    data[score_col] = data.apply(
        lambda row: calculate_url_title_match_score(row[title_col], row[url_col])
        if pd.isnull(row[score_col])
        else row[score_col],
        axis=1,
    )
    return data


def calculate_url_title_match_score(title: Any, url: Any) -> Any:
    if pd.notnull(title) and pd.notnull(url):
        tSet = title.split(" ")
        txtURL = urlparse(url).netloc.split(".")[:-1] + [
            i for i in urlparse(url).path.split("/") if i != ""
        ]
        txtURL = [i for i in txtURL if i != "www"]
        txtURL = ".".join(txtURL)

        score = 0.0
        baseScore = 100 / len(txtURL) if len(txtURL) > 0 else 0

        for element in tSet:
            if element in txtURL:
                n = len(element)
                score += baseScore * n
                txtURL = txtURL.replace(element, "")
                if score > 99.9:
                    score = 100
        return score
    return np.nan


# ---------------------------------------------------------------------------
# Row-level twins
# ---------------------------------------------------------------------------
#
# These are a convenience for single-URL inference, which does not need a DataFrame apply
# to compute one value. They are NOT a second implementation: the DataFrame functions above
# remain the authority, and a property test asserts each twin agrees with its DataFrame
# counterpart on a one-row frame across a 5,000-URL sample. If the two ever disagree, the
# twin is what is wrong.


def url_length(url: str | None) -> float | None:
    return len(str(url)) if url is not None and pd.notnull(url) else None


def domain_of(url: str | None) -> str | None:
    return urlparse(url).netloc if url is not None and pd.notnull(url) else None


def domain_length(domain: str | None) -> float | None:
    return len(str(domain)) if domain is not None and pd.notnull(domain) else None


def is_domain_ip(domain: str | None) -> int | None:
    if domain is None or not pd.notnull(domain):
        return None
    if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain):
        return 1 if all(0 <= int(p) <= 255 for p in domain.split(".")) else 0
    return 0


def tld_of(domain: str | None) -> str | None:
    if domain is not None and pd.notnull(domain):
        parts = domain.split(".")
        if len(parts) > 1:
            return parts[-1]
    return None


def char_continuation_rate(url: str | None) -> float | None:
    if url is None or not pd.notnull(url):
        return None
    total = sum(len(s) for s in re.findall(r"[a-zA-Z0-9_]+", url))
    return total / len(url) if len(url) > 0 else None


def url_char_prob(url: str | None, char_prob: dict[str, float]) -> float | None:
    if url is None or not pd.notnull(url):
        return None
    total = sum(char_prob.get(c, 0) for c in url.lower() if c.isalnum())
    return total / len(url) if len(url) > 0 else None


def tld_length(tld: str | None) -> float | None:
    return len(str(tld)) if tld is not None and pd.notnull(tld) else None


def no_of_subdomains(domain: str | None) -> float | None:
    if domain is None or not pd.notnull(domain):
        return None
    parts = domain.split(".")
    return len(parts) - 2 if len(parts) > 2 else 0


def has_obfuscation(url: str | None) -> int:
    return detect_advanced_obfuscation(url)


def no_of_obfuscated_chars(url: str | None) -> float | None:
    if url is None or not pd.notnull(url):
        return None
    return len(re.findall(r"%[0-9a-fA-F]{2}", url)) + url.count("@")


def no_of_letters(url: str | None) -> float | None:
    return sum(c.isalpha() for c in url) if url is not None and pd.notnull(url) else None


def no_of_digits(url: str | None) -> float | None:
    return sum(c.isdigit() for c in url) if url is not None and pd.notnull(url) else None


def no_of_equals(url: str | None) -> float | None:
    return sum(c == "=" for c in url) if url is not None and pd.notnull(url) else None


def no_of_qmark(url: str | None) -> float | None:
    return sum(c == "?" for c in url) if url is not None and pd.notnull(url) else None


def no_of_ampersand(url: str | None) -> float | None:
    return sum(c == "&" for c in url) if url is not None and pd.notnull(url) else None


def no_of_special_chars(url: str | None) -> float | None:
    if url is None or not pd.notnull(url):
        return None
    return sum(c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" for c in url)


def is_https(url: str | None) -> int | None:
    if url is None or not pd.notnull(url):
        return None
    return 1 if "https://" in url else 0
