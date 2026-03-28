"""
Structural rule engine – cheap deterministic checks applied BEFORE the ML model.

Each rule returns a tuple: (result: str | None, confidence: float | None)
- result is "PASS", "FLAG", or None (undecided)
- confidence is a heuristic 0.0–1.0 score

A None result means the rule layer cannot confidently decide and the template
should fall through to the ML model.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Internal patterns used by rules
# ---------------------------------------------------------------------------

# Sender prefix: "Brand: message" or "Brand - message" (short token before colon/dash)
_RE_SENDER_PREFIX = re.compile(
    r"^[A-Za-z][A-Za-z0-9 \-&\.]{1,40}(?::|–|-)\s",
)

# Signature suffix: "- Team X", "- CompanyName", "- Brand Support"
_RE_SIGNATURE_SUFFIX = re.compile(
    r"[-–]\s+[A-Z][A-Za-z0-9 &\.]{1,40}\s*$",
)

# Organization-like token: capitalized word that looks like a brand name.
# Includes both TitleCase (Daraz, Pathao) and camelCase (bKash, eBay).
# Minimum 3 chars.
_GENERIC_WORDS = {
    "YOUR", "OTP", "THE", "CODE", "PIN", "SMS", "PLEASE",
    "DEAR", "SIR", "MADAM", "CLICK", "HERE", "THANK", "YOU",
    "GET", "USE", "NEW", "OLD", "THIS", "THAT", "FROM", "FOR",
    "PAYMENT", "AMOUNT", "ORDER", "VISIT", "LINK", "LOGIN",
    "ACCOUNT", "USER", "PASS", "VALID", "EXPIRE", "VERIFY",
    "CONFIRM", "COMPLETE", "DONE", "PENDING", "SUCCESS", "FAILED",
    "RECEIVED", "SENT",
}

# TitleCase brand: starts uppercase, at least 3 chars total
_RE_CAPITALIZED_BRAND = re.compile(
    r"\b([A-Z][A-Za-z0-9]{2,})\b"
)

# camelCase brand: starts lowercase but contains internal uppercase (e.g. bKash, eBay)
_RE_CAMELCASE_BRAND = re.compile(
    r"\b([a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)\b"
)

# Greeting with app/service mention: "Hi from Brand" / "Welcome to Brand"
# Accepts both TitleCase and camelCase words after the preposition
_RE_GREETING_BRAND = re.compile(
    r"\b(?:from|to|by|via|on|at)\s+([A-Za-z][A-Za-z0-9]{2,})\b"
)

# URL / domain already detected by normalizer → placeholder present
_RE_URL_PLACEHOLDER = re.compile(r"\{URL\}|\{DOMAIN\}")


# ---------------------------------------------------------------------------
# Public rule function
# ---------------------------------------------------------------------------

RuleResult = tuple[Optional[str], Optional[float]]


def apply_rules(template_text: str) -> RuleResult:
    """
    Apply structural rules to *template_text*.

    Returns:
        (result, confidence) where result is "PASS", "FLAG", or None.
        None means the rule layer is undecided; fall through to ML model.
    """

    # Rule 1: Contains a URL or domain placeholder → sender is likely identifiable
    if _RE_URL_PLACEHOLDER.search(template_text):
        return "PASS", 0.85

    # Rule 2: Sender prefix pattern "BrandName: ..."
    if _RE_SENDER_PREFIX.match(template_text):
        return "PASS", 0.95

    # Rule 3: Signature suffix "- BrandName" / "- Team Support"
    if _RE_SIGNATURE_SUFFIX.search(template_text):
        return "PASS", 0.90

    # Rule 4: Greeting/context brand mention "from Daraz", "via bKash"
    greeting_match = _RE_GREETING_BRAND.search(template_text)
    if greeting_match:
        token = greeting_match.group(1)
        # Brand token must have at least one uppercase character (rules out
        # plain common words like "your", "the", "payment")
        has_upper = any(c.isupper() for c in token)
        if has_upper and token.upper() not in _GENERIC_WORDS and len(token) >= 3:
            return "PASS", 0.85

    # Rule 5: Look for a capitalized non-generic token that could be a brand.
    # Check both TitleCase and camelCase patterns.
    # Strip normalization placeholders first so {NUM}, {URL} etc. don't match.
    text_no_placeholders = re.sub(r"\{[A-Z]+\}", "", template_text)
    caps_hits = _RE_CAPITALIZED_BRAND.findall(text_no_placeholders)
    camel_hits = _RE_CAMELCASE_BRAND.findall(text_no_placeholders)
    all_hits = caps_hits + camel_hits
    non_generic = [
        t for t in all_hits
        if t.upper() not in _GENERIC_WORDS and len(t) >= 3
    ]
    if len(non_generic) >= 1:
        # At least one brand-like token found
        return "PASS", 0.75

    # Rule 6: Template is extremely generic – only placeholders and common words
    # If after removing all placeholders the meaningful word count is very low
    # and no brand signal detected, lean toward FLAG but with low confidence
    stripped = re.sub(r"\{[A-Z]+\}", "", template_text).strip()
    word_count = len(stripped.split())
    if word_count <= 3:
        # Very sparse template – likely a pure "Your OTP is {NUM}" style message
        return "FLAG", 0.70

    # Rule layer undecided – defer to ML model
    return None, None
