# ruff: noqa
# type: ignore
# fmt: off
"""FROZEN VERBATIM COPY OF THE ORIGINAL NOTEBOOK'S FEATURE FUNCTIONS.

DO NOT EDIT. DO NOT REFORMAT. DO NOT LINT. DO NOT "FIX".

This module is the oracle. The production implementations in
``phishguard/features/url_features.py`` are asserted bit-identical to these functions over
a 2,000-row sample, and that assertion is the only evidence that the port preserved what
the training data means. Changing a character here changes the oracle, which would let a
real port defect pass the test that exists to catch it.

The functions are copied from the notebook cells noted above each one, with exactly three
removals, all of which are non-semantic:

  1. The trailing demonstration statements each cell appended after the definition
     (``X_train = fill_url_length(X_train, ...)`` and the DataFrame echo).
  2. Nothing else. ``print()`` calls inside function bodies are RETAINED, because removing
     them would be an edit, and they are harmless under pytest's captured stdout.
  3. Nothing else.

Known defects are preserved deliberately and are documented rather than repaired:

  - ``fill_tld_legitimate_prob`` computes its own imputation statistics from whatever
    frame it is handed. That is precisely the train/serve skew the production port exists
    to eliminate, and it must remain here so the port can be compared against it.
  - ``detect_advanced_obfuscation`` rule 4 tests whether the *reversed* URL matches
    ``^[a-zA-Z0-9.\\-]+$``. A string matches that pattern exactly when its reversal does,
    so the rule fires for any URL made only of letters, digits, dots and hyphens -- which
    includes every bare domain with no scheme. Almost certainly not the intent. Preserved.
  - ``fill_has_obfuscation``'s final coercion is ``lambda x: 1 if x == True else 0``, which
    maps any pre-existing non-1 value (including a legitimate 0) through the same branch.
    Preserved.
"""

import base64
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd


# --- cell 22 -----------------------------------------------------------------
def fill_url_length(data, url_col='URL', url_length_col='URLLength'):
    data[url_length_col] = data.apply(
        lambda row: len(str(row[url_col])) if pd.isnull(row[url_length_col]) and pd.notnull(row[url_col]) else row[url_length_col],
        axis=1
    )
    return data


# --- cell 25 -----------------------------------------------------------------
def fill_domain(data, url_col='URL', domain_col='Domain'):
    def get_domain(url):
        return urlparse(url).netloc if pd.notnull(url) else None

    data[domain_col] = data.apply(
        lambda row: get_domain(row[url_col]) if pd.isnull(row[domain_col]) and pd.notnull(row[url_col]) else row[domain_col],
        axis=1
    )
    return data


# --- cell 28 -----------------------------------------------------------------
def fill_domain_length(data, domain_col='Domain', domain_length_col='DomainLength'):
    data[domain_length_col] = data.apply(
        lambda row: len(str(row[domain_col]))
        if pd.isnull(row[domain_length_col]) and pd.notnull(row[domain_col])
        else row[domain_length_col],
        axis=1
    )
    return data


# --- cell 31 -----------------------------------------------------------------
def fill_is_domain_ip(data, domain_col='Domain', is_domain_ip_col='IsDomainIP'):
    def is_ipaddress(domain):
        if pd.notnull(domain):
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            if re.match(ip_pattern, domain):
                parts = domain.split('.')
                return all(0 <= int(part) <= 255 for part in parts)
        return False

    data[is_domain_ip_col] = data.apply(
        lambda row: (1 if is_ipaddress(row[domain_col]) else 0)
        if pd.isnull(row[is_domain_ip_col]) and pd.notnull(row[domain_col])
        else row[is_domain_ip_col],
        axis=1
    )
    return data


# --- cell 34 -----------------------------------------------------------------
def fill_tld(data, domain_col='Domain', tld_col='TLD'):
    def generate_tld(domain):
        if pd.notnull(domain):
            parts = domain.split('.')
            if len(parts) > 1:
                return parts[-1]
        return np.nan

    data[tld_col] = data.apply(
        lambda row: generate_tld(row[domain_col])
        if pd.isnull(row[tld_col])
        else row[tld_col],
        axis=1
    )
    return data


# --- cell 37 -----------------------------------------------------------------
def fill_char_continuation_rate(data, url_col='URL', char_rate_col='CharContinuationRate'):
    def generate_char_continuation_rate(url):
        if pd.notnull(url):
            sequences = re.findall(r'[a-zA-Z0-9_]+', url)
            total_sequence_length = sum(len(seq) for seq in sequences)
            total_url_length = len(url)
            return total_sequence_length / total_url_length if total_url_length > 0 else np.nan
        return np.nan

    data[char_rate_col] = data.apply(
        lambda row: generate_char_continuation_rate(row[url_col])
        if pd.isnull(row[char_rate_col]) else row[char_rate_col],
        axis=1
    )
    return data


# --- cell 40 -----------------------------------------------------------------
def fill_tld_legitimate_prob(data, tld_col='TLD', tld_prob_col='TLDLegitimateProb', is_domain_ip_col='IsDomainIP'):

    skewness = data[tld_prob_col].skew()
    if skewness > 1 or skewness < -1:
        global_fill_value = data[tld_prob_col].median()
        print("Uses median for imputation because distribution is skewed.")
    else:
        global_fill_value = data[tld_prob_col].mean()
        print("Uses the mean for imputation because of the normal distribution.")

    print(f"Global fill value: {global_fill_value}")


    tld_prob_mean = data.groupby(tld_col)[tld_prob_col].mean()


    def fill_tld_legit_prob(row):
        if pd.isnull(row[tld_prob_col]):
            if row[is_domain_ip_col] == 1:
                return 0
            if pd.notnull(row[tld_col]) and row[tld_col] in tld_prob_mean.index:  # TLD not found
                return tld_prob_mean[row[tld_col]]
            if pd.isnull(row[tld_col]) and pd.isnull(row[is_domain_ip_col]):  # TLD dan IsDomainIP not found
                return global_fill_value
        return row[tld_prob_col]
    data[tld_prob_col] = data.apply(fill_tld_legit_prob, axis=1)
    data[tld_prob_col] = data[tld_prob_col].fillna(global_fill_value)

    return data


# --- cell 43 -----------------------------------------------------------------
# Function to calculate character probabilities
def calculate_char_prob(df, url_col):
    char_count = {}
    total_chars = 0
    for url in df[url_col].dropna():
        for char in url.lower():
            if char.isalnum():  # Only consider alphanumeric characters
                char_count[char] = char_count.get(char, 0) + 1
                total_chars += 1
    return {char: count / total_chars for char, count in char_count.items()}

# Function to calculate URLCharProb
def fill_url_char_prob(data, url_col='URL', char_prob_col='URLCharProb', char_prob=None):
    if char_prob is None:
        raise ValueError("Character probabilities (`char_prob`) must be provided.")

    def calculate_url_char_prob(url):
        if pd.notnull(url):
            total_prob = sum(char_prob.get(char, 0) for char in url.lower() if char.isalnum())
            n = len(url)
            return total_prob / n if n > 0 else np.nan
        return np.nan

    data[char_prob_col] = data.apply(
        lambda row: calculate_url_char_prob(row[url_col])
        if pd.isnull(row[char_prob_col]) else row[char_prob_col],
        axis=1
    )
    return data


# --- cell 46 -----------------------------------------------------------------
def fill_tld_length(data, tld_col='TLD', tld_length_col='TLDLength'):
    def calculate_tld_length(tld):
        if pd.notnull(tld):
            return len(str(tld))
        return np.nan

    data[tld_length_col] = data.apply(
        lambda row: calculate_tld_length(row[tld_col])
        if pd.isnull(row[tld_length_col]) else row[tld_length_col],
        axis=1
    )
    return data


# --- cell 49 -----------------------------------------------------------------
def fill_no_of_subdomains(data, domain_col='Domain', subdomain_col='NoOfSubDomain'):
    def calculate_no_of_subdomains(domain):
        if pd.notnull(domain):
            parts = domain.split('.')
            return len(parts) - 2 if len(parts) > 2 else 0
        return np.nan

    data[subdomain_col] = data.apply(
        lambda row: calculate_no_of_subdomains(row[domain_col])
        if pd.isnull(row[subdomain_col]) else row[subdomain_col],
        axis=1
    )
    return data


# --- cell 52 -----------------------------------------------------------------
def fill_has_obfuscation(data, url_col='URL', obfuscation_col='HasObfuscation'):
    def detect_advanced_obfuscation(url):
        if pd.notnull(url):
            if len(re.findall(r'[-_]', url)) > 3:  # Rule 1: Too many special characters
                return 1
            if re.search(r'[a-zA-Z]+\d+|\d+[a-zA-Z]+', url):  # Rule 2: Mixed alphanumeric patterns
                return 1
            if len(url) % 4 == 0 and re.match(r'^[A-Za-z0-9+/]*={0,2}$', url):  # Rule 3: Base64 encoded strings
                try:
                    base64.b64decode(url, validate=True)
                    return 1
                except Exception:
                    pass
            reversed_url = url[::-1]  # Rule 4: Reversed strings
            if re.match(r'^[a-zA-Z0-9.\-]+$', reversed_url):
                return 1
            if not re.search(r'[a-zA-Z]{3,}', url):  # Rule 5: Randomized strings without meaningful words
                return 1
            return 0  # No obfuscation detected
        return 0  # Missing values treated as no obfuscation

    data[obfuscation_col] = data.apply(
        lambda row: detect_advanced_obfuscation(row[url_col])
        if pd.isnull(row[obfuscation_col]) and pd.notnull(row[url_col])
        else row[obfuscation_col],
        axis=1
    )
    data[obfuscation_col] = data[obfuscation_col].apply(lambda x: 1 if x == True else 0)
    return data


# --- cell 55 -----------------------------------------------------------------
def fill_no_of_obfuscated_characters(data, url_col='URL', obf_char_col='NoOfObfuscatedChar'):
    def count_obfuscated_characters(url):
        if pd.notnull(url):
            hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', url))  # Count %XX hexadecimal patterns
            at_count = url.count('@')  # Count @ symbol
            return hex_count + at_count  # Total obfuscated characters
        return np.nan

    data[obf_char_col] = data.apply(
        lambda row: count_obfuscated_characters(row[url_col])
        if pd.isnull(row[obf_char_col]) else row[obf_char_col],
        axis=1
    )
    return data


# --- cell 58 -----------------------------------------------------------------
def fill_obfuscation_ratio(data, obf_char_col='NoOfObfuscatedChar', url_length_col='URLLength', obf_ratio_col='ObfuscationRatio'):
    def calculate_obfuscation_ratio(no_of_obfchar, url_length):
        if pd.notnull(no_of_obfchar) and pd.notnull(url_length) and url_length > 0:
            return no_of_obfchar / url_length
        return np.nan

    data[obf_ratio_col] = data.apply(
        lambda row: calculate_obfuscation_ratio(row[obf_char_col], row[url_length_col])
        if pd.isnull(row[obf_ratio_col]) else row[obf_ratio_col],
        axis=1
    )
    return data


# --- cell 61 -----------------------------------------------------------------
def fill_no_of_letters_in_url(data, url_col='URL', letters_col='NoOfLettersInURL'):
    def calculate_no_of_letters(url):
        if pd.notnull(url):
            return sum(c.isalpha() for c in url)
        return np.nan

    data[letters_col] = data.apply(
        lambda row: calculate_no_of_letters(row[url_col])
        if pd.isnull(row[letters_col]) else row[letters_col],
        axis=1
    )
    return data


# --- cell 64 -----------------------------------------------------------------
def fill_letter_ratio_in_url(data, letters_col='NoOfLettersInURL', url_length_col='URLLength', ratio_col='LetterRatioInURL'):
    def calculate_letter_ratio(no_of_letters, url_length):
        if pd.notnull(no_of_letters) and pd.notnull(url_length) and url_length > 0:
            return no_of_letters / url_length
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_letter_ratio(row[letters_col], row[url_length_col])
        if pd.isnull(row[ratio_col]) else row[ratio_col],
        axis=1
    )
    return data


# --- cell 67 -----------------------------------------------------------------
def fill_no_of_digits_in_url(data, url_col='URL', digits_col='NoOfDegitsInURL'):
    def calculate_no_of_digits(url):
        if pd.notnull(url):
            return sum(c.isdigit() for c in url)
        return np.nan

    data[digits_col] = data.apply(
        lambda row: calculate_no_of_digits(row[url_col])
        if pd.isnull(row[digits_col]) else row[digits_col],
        axis=1

    )
    return data


# --- cell 70 -----------------------------------------------------------------
def fill_digits_ratio_in_url(data, url_col='URL', ratio_col='DegitRatioInURL'):
    def calculate_digits_ratio(url):
        if pd.notnull(url) and len(url) > 0:
            return sum(c.isdigit() for c in url) / len(url)
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_digits_ratio(row[url_col])
        if pd.isnull(row[ratio_col]) else row[ratio_col],
        axis=1
    )
    return data


# --- cell 73 -----------------------------------------------------------------
def fill_no_of_equals_in_url(data, url_col='URL', equals_col='NoOfEqualsInURL'):
    def calculate_no_of_equals(url):
        if pd.notnull(url):
            return sum(c == "=" for c in url)
        return np.nan

    data[equals_col] = data.apply(
        lambda row: calculate_no_of_equals(row[url_col])
        if pd.isnull(row[equals_col]) else row[equals_col],
        axis=1
    )
    return data


# --- cell 76 -----------------------------------------------------------------
def fill_no_of_qmark_in_url(data, url_col='URL', qmark_col='NoOfQMarkInURL'):
    def calculate_no_of_qmark(url):
        if pd.notnull(url):
            return sum(c == "?" for c in url)
        return np.nan

    data[qmark_col] = data.apply(
        lambda row: calculate_no_of_qmark(row[url_col])
        if pd.isnull(row[qmark_col]) else row[qmark_col],
        axis=1
    )
    return data


# --- cell 79 -----------------------------------------------------------------
def fill_no_of_ampersand_in_url(data, url_col='URL', ampersand_col='NoOfAmpersandInURL'):
    def calculate_no_of_ampersand(url):
        if pd.notnull(url):
            return sum(c == "&" for c in url)
        return np.nan

    data[ampersand_col] = data.apply(
        lambda row: calculate_no_of_ampersand(row[url_col])
        if pd.isnull(row[ampersand_col]) else row[ampersand_col],
        axis=1
    )
    return data


# --- cell 82 -----------------------------------------------------------------
def fill_no_of_special_chars_in_url(data, url_col='URL', special_chars_col='NoOfOtherSpecialCharsInURL'):
    def calculate_no_of_specials(url):
        if pd.notnull(url):
            return sum(c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" for c in url)
        return np.nan

    data[special_chars_col] = data.apply(
        lambda row: calculate_no_of_specials(row[url_col])
        if pd.isnull(row[special_chars_col]) else row[special_chars_col],
        axis=1
    )
    return data


# --- cell 85 -----------------------------------------------------------------
def fill_specials_ratio_in_url(data, url_col='URL', ratio_col='SpacialCharRatioInURL'):
    def calculate_specials_ratio(url):
        if pd.notnull(url) and len(url) > 0:
            return sum(c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" for c in url) / len(url)
        return np.nan

    data[ratio_col] = data.apply(
        lambda row: calculate_specials_ratio(row[url_col])
        if pd.isnull(row[ratio_col]) else row[ratio_col],
        axis=1
    )
    return data


# --- cell 88 -----------------------------------------------------------------
def fill_is_https(data, url_col='URL', https_col='IsHTTPS'):
    def calculate_ishttps(url):
        if pd.notnull(url):
            return 1 if "https://" in url else 0
        return np.nan

    data[https_col] = data.apply(
        lambda row: calculate_ishttps(row[url_col])
        if pd.isnull(row[https_col]) else row[https_col],
        axis=1
    )
    return data


# --- cell 91 -----------------------------------------------------------------
def fill_has_title(data, title_col='Title', has_title_col='HasTitle'):
    data[has_title_col] = data.apply(
        lambda row: row[has_title_col]
        if not pd.isnull(row[has_title_col])
        else (1 if not pd.isnull(row[title_col]) else row[has_title_col]),
        axis=1
    )
    return data


# --- cell 94 -----------------------------------------------------------------
def fill_domain_title_match_score(data, title_col='Title', domain_col='Domain', score_col='DomainTitleMatchScore'):
    def calculate_match_score(title, domain):
        tSet = title.split(" ") if pd.notnull(title) else []
        txtDomain = domain.split(".")[:-1] if pd.notnull(domain) else []
        txtDomain = [i for i in txtDomain if i != "www"]
        txtDomain = ".".join(txtDomain)

        score = 0
        baseScore = 100 / len(txtDomain) if len(txtDomain) > 0 else 0

        for element in tSet:
            if element in txtDomain:
                n = len(element)
                score += baseScore * n
                txtDomain = txtDomain.replace(element, "")
                if score > 99.9:
                    score = 100
        return score

    data[score_col] = data.apply(
        lambda row: calculate_match_score(row[title_col], row[domain_col])
        if pd.isnull(row[score_col])
        and pd.notnull(row[title_col])
        and pd.notnull(row[domain_col])
        else row[score_col],
        axis=1
    )
    return data


# --- cell 97 -----------------------------------------------------------------
def fill_url_title_match_score(data, title_col='Title', url_col='URL', score_col='URLTitleMatchScore'):
    def calculate_url_title_match_score(title, url):
        if pd.notnull(title) and pd.notnull(url):
            tSet = title.split(" ")
            txtURL = (
                urlparse(url).netloc.split(".")[:-1] +
                [i for i in urlparse(url).path.split("/") if i != ""]
            )
            txtURL = [i for i in txtURL if i != "www"]
            txtURL = ".".join(txtURL)

            score = 0
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

    data[score_col] = data.apply(
        lambda row: calculate_url_title_match_score(row[title_col], row[url_col])
        if pd.isnull(row[score_col]) else row[score_col],
        axis=1
    )
    return data


# --- cell 102 ----------------------------------------------------------------
def fill_numerical_columns(data, numerical_columns, skewness_threshold=3):
    for col in numerical_columns:
        if col in data.columns:
            skewness = data[col].skew()
            if skewness > skewness_threshold or skewness < -skewness_threshold:
                fill_value = data[col].median()
                print(f"Column {col}: Using median for imputation because distribution is highly skewed.")
            else:
                fill_value = data[col].mean()
                print(f"Column {col}: Using mean for imputation because distribution is approximately normal.")
            data[col] = data[col].fillna(fill_value)
    print("Missing values after imputation:")
    print(data[numerical_columns].isnull().sum())
    return data


# --- cell 104 ----------------------------------------------------------------
def fill_categorical_columns(data, categorical_columns, initial_group_by):
    group_by_columns = initial_group_by.copy()

    for col in categorical_columns:
        mode_values = (
            data.groupby(group_by_columns)[col]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        )

        global_mode = data[col].mode()[0] if not data[col].mode().empty else None

        def fill_value(row):
            if pd.isnull(row[col]):
                group_key = tuple(row[gb] for gb in group_by_columns)
                return mode_values.get(group_key, global_mode)
            return row[col]

        data[col] = data.apply(fill_value, axis=1)

        group_by_columns.append(col)

    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            global_mode = data[col].mode()[0] if not data[col].mode().empty else None
            data[col] = data[col].fillna(global_mode)

    return data
