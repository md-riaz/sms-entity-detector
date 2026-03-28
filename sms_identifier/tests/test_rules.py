"""
Tests for the rule engine.
"""

import sys
import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from app.rules import apply_rules


class TestApplyRules:
    # ---- PASS cases ----

    def test_url_placeholder_passes(self):
        result, confidence = apply_rules("Click {URL} to verify your account")
        assert result == "PASS"
        assert confidence is not None and confidence > 0.5

    def test_domain_placeholder_passes(self):
        result, confidence = apply_rules("Visit {DOMAIN} to continue")
        assert result == "PASS"

    def test_sender_prefix_passes(self):
        result, confidence = apply_rules("Daraz: Your order {NUM} is confirmed")
        assert result == "PASS"
        assert confidence is not None and confidence >= 0.9

    def test_bkash_prefix_passes(self):
        result, confidence = apply_rules("bKash: OTP is {NUM}")
        assert result == "PASS"

    def test_signature_suffix_passes(self):
        result, confidence = apply_rules("Your account is updated - Grameenphone")
        assert result == "PASS"

    def test_via_brand_passes(self):
        result, confidence = apply_rules("Sent via bKash to confirm")
        assert result == "PASS"

    def test_from_brand_passes(self):
        result, confidence = apply_rules("Notification from Daraz")
        assert result == "PASS"

    def test_capitalized_brand_token(self):
        result, confidence = apply_rules("Pathao OTP is {NUM}")
        assert result == "PASS"

    # ---- FLAG cases ----

    def test_generic_otp_flags(self):
        # "Your OTP is {NUM}" – after normalization only generic words remain
        result, confidence = apply_rules("Your OTP is {NUM}")
        # Rule 6: only generic words / placeholders → FLAG
        assert result == "FLAG"

    def test_payment_received_flags(self):
        result, confidence = apply_rules("Payment received successfully")
        # Few words, no brand signal
        assert result == "FLAG"

    # ---- Undecided cases (None) ----

    def test_ambiguous_returns_none(self):
        # A sentence that has words but no clear brand signal
        # and isn't sparse enough to trigger FLAG rule 6
        result, confidence = apply_rules(
            "Please verify your identity using the provided link and code"
        )
        # This may return None (model needed) or FLAG depending on word count
        # Key assertion: result is either None or FLAG, never PASS
        assert result in (None, "FLAG")

    # ---- Confidence range ----

    def test_confidence_is_float_or_none(self):
        result, confidence = apply_rules("Click {URL} to verify")
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0

    def test_url_confidence_high(self):
        _, confidence = apply_rules("Reset password at {URL}")
        assert confidence is not None and confidence >= 0.8
