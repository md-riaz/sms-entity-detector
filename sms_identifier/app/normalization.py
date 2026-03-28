"""
SMS normalization – converts raw SMS text into a stable template by
replacing dynamic values with typed placeholders.

Replacement order matters: more-specific patterns must run before
less-specific ones (e.g., URLs before bare numbers).
"""

import hashlib
import re

# ---------------------------------------------------------------------------
# Compiled regex patterns (ordered from most- to least-specific)
# ---------------------------------------------------------------------------

# Full URLs (http/https/ftp)
_RE_URL = re.compile(
    r"https?://[^\s]+|ftp://[^\s]+",
    re.IGNORECASE,
)

# Domain-like tokens that are NOT full URLs but look like domains (e.g., company.com)
_RE_DOMAIN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"(?:com|net|org|io|app|co|in|bd|uk|us|info|biz|ly|me|pk|sg|au)\b",
    re.IGNORECASE,
)

# Email addresses (before bare number patterns)
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# ISO-style dates: 2024-01-31 / 31-01-2024 / 31/01/2024 / Jan 31, 2024
_RE_DATE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
    re.IGNORECASE,
)

# Times: 12:30, 12:30:59, 3:45 PM
_RE_TIME = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?\b",
)

# Long alphanumeric IDs: at least 6 chars, mix of letters and digits
# (order before bare _RE_NUM so it catches things like "TXN12345" first)
_RE_ID = re.compile(
    r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]{6,}\b",
)

# Pure numbers (integers and decimals, including with commas e.g. 1,234.56)
_RE_NUM = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\b",
)

# Placeholders already inserted – protect them from double-replacement
_PLACEHOLDER = re.compile(r"\{(?:URL|DOMAIN|EMAIL|DATE|TIME|ID|NUM)\}")


def normalize(text: str) -> str:
    """
    Replace dynamic values in *text* with stable typed placeholders.

    Returns the normalized template string.
    Normalizes whitespace and strips leading/trailing spaces at the end.
    """
    t = text

    # 1. Full URLs
    t = _RE_URL.sub("{URL}", t)

    # 2. Email addresses (before domain so "user@brand.com" → {EMAIL} not "user@{DOMAIN}")
    t = _RE_EMAIL.sub("{EMAIL}", t)

    # 3. Domain-like tokens (bare domains, not already replaced)
    t = _RE_DOMAIN.sub("{DOMAIN}", t)

    # 4. Dates (before time/num to avoid partial overlap)
    t = _RE_DATE.sub("{DATE}", t)

    # 5. Times
    t = _RE_TIME.sub("{TIME}", t)

    # 6. Long alphanumeric IDs (uppercase + digit mix, 6+ chars)
    t = _RE_ID.sub("{ID}", t)

    # 7. Plain numbers
    t = _RE_NUM.sub("{NUM}", t)

    # Collapse multiple spaces / newlines to single space
    t = re.sub(r"\s+", " ", t).strip()

    return t


def compute_hash(template_text: str) -> str:
    """Return a SHA-1 hex digest for the given template string."""
    return hashlib.sha1(template_text.encode("utf-8")).hexdigest()
