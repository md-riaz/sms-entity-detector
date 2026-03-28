"""
Tests for the normalization module.
"""

import sys
import os

# Ensure package root is importable
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from app.normalization import normalize, compute_hash


class TestNormalize:
    def test_url_replaced(self):
        assert normalize("Visit https://example.com now") == "Visit {URL} now"

    def test_http_url(self):
        assert normalize("Go to http://brand.io/path?q=1") == "Go to {URL}"

    def test_domain_replaced(self):
        result = normalize("Go to brand.com for help")
        assert "{DOMAIN}" in result

    def test_email_replaced(self):
        result = normalize("Contact us at support@brand.com")
        assert "{EMAIL}" in result

    def test_otp_number_replaced(self):
        result = normalize("Your OTP is 123456")
        assert "123456" not in result
        assert "{NUM}" in result

    def test_decimal_number_replaced(self):
        result = normalize("Amount: 1,234.56 BDT")
        assert "1,234.56" not in result

    def test_date_replaced(self):
        result = normalize("Expires on 2024-12-31")
        assert "{DATE}" in result
        assert "2024" not in result

    def test_time_replaced(self):
        result = normalize("Meeting at 10:30 AM")
        assert "{TIME}" in result
        assert "10:30" not in result

    def test_alphanumeric_id_replaced(self):
        result = normalize("TXN123ABC order confirmed")
        assert "{ID}" in result
        assert "TXN123ABC" not in result

    def test_whitespace_normalized(self):
        result = normalize("Hello   World  ")
        assert result == "Hello World"

    def test_static_text_unchanged(self):
        result = normalize("Your payment was received successfully")
        assert result == "Your payment was received successfully"

    def test_brand_name_preserved(self):
        result = normalize("bKash OTP is 123456")
        assert "bKash" in result

    def test_brand_with_colon_preserved(self):
        result = normalize("Daraz: Order confirmed for 150")
        assert "Daraz:" in result

    def test_multiple_replacements(self):
        raw = "Your OTP 987654 expires at 12:30 PM on 2024-06-01"
        result = normalize(raw)
        assert "987654" not in result
        assert "12:30" not in result
        assert "2024-06-01" not in result

    def test_empty_string(self):
        assert normalize("") == ""


class TestComputeHash:
    def test_same_input_same_hash(self):
        h1 = compute_hash("Hello World")
        h2 = compute_hash("Hello World")
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = compute_hash("Hello World")
        h2 = compute_hash("Hello World!")
        assert h1 != h2

    def test_hash_is_hex(self):
        h = compute_hash("test")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_sha1(self):
        h = compute_hash("test")
        assert len(h) == 40  # SHA-1 produces 40 hex chars

    def test_normalized_template_hash_stable(self):
        # Two SMS with same template but different OTPs should produce same hash
        t1 = normalize("Your OTP is 111111")
        t2 = normalize("Your OTP is 999999")
        assert t1 == t2
        assert compute_hash(t1) == compute_hash(t2)
