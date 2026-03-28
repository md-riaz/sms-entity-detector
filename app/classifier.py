"""
Classifier wrapper around a pretrained HuggingFace NER model.

Design goal: this module is the only place that knows about the ML model.
Swapping models or adding a second classifier requires changes only here.

Model used: Davlan/xlm-roberta-base-ner-hrl
  – multilingual NER, supports ORG, PER, LOC entity types
  – entity labels: B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC, O
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline – only initialised once on first classify() call
# or explicitly via load_model().
_pipeline = None
_model_loaded: bool = False
_model_error: Optional[str] = None

# Transactional keywords that strongly suggest an app/service sender context.
TRANSACTIONAL_HINTS = {
    'otp',
    'security code',
    'verification code',
    'verify',
    'account',
    'login',
    'password',
    'pin',
    'platform',
    'code',
}


def _extract_rule_entity(template_text: str) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Try to extract a sender-like entity from template_text for rule-based decisions.
    Returns (entity, label, score) or (None, None, None).
    """
    import re
    patterns = [
        r'for\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)',
        r'^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+OTP',
        r'^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+Platform',
        r'([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+OTP',  # "Proiojon OTP"
        r'([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+Platform',  # "Proiojon Platform"
        r'for\s+([A-Za-z][A-Za-z0-9]+(?:\s+[A-Za-z][A-Za-z0-9]+)*)',  # "for Proiojon" (lowercase allowed)
    ]
    for pattern in patterns:
        match = re.search(pattern, template_text)
        if match:
            return match.group(1), "ORG", 0.75
    return None, None, None


def load_model() -> bool:
    """
    Load the NER pipeline.
    Returns True on success, False on failure.
    Called once at application startup (or lazily on first inference).
    """
    global _pipeline, _model_loaded, _model_error

    from app.config import MODEL_ENABLED, NER_MODEL_NAME

    if not MODEL_ENABLED:
        logger.warning("Model loading disabled via MODEL_ENABLED=false")
        _model_error = "Model disabled"
        return False

    if _model_loaded:
        return True

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        logger.info("Loading NER model: %s", NER_MODEL_NAME)
        _pipeline = hf_pipeline(
            "ner",
            model=NER_MODEL_NAME,
            aggregation_strategy="simple",  # merge B-/I- tokens automatically
            device=-1,  # CPU; set to 0 for GPU
        )
        _model_loaded = True
        _model_error = None
        logger.info("NER model loaded successfully")
        return True
    except Exception as exc:
        _model_loaded = False
        _model_error = str(exc)
        logger.error("Failed to load NER model: %s", exc)
        return False


def is_model_ready() -> bool:
    return _model_loaded


def model_error() -> Optional[str]:
    return _model_error


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

ClassifyResult = tuple[str, Optional[float], Optional[str], Optional[str], Optional[float]]  # (result, confidence, entity, label, score)


def _looks_transactional(template_text: str) -> bool:
    lowered = template_text.lower()
    return any(hint in lowered for hint in TRANSACTIONAL_HINTS)


ClassifyResult = tuple[str, Optional[float], Optional[str], Optional[str], Optional[float]]  # (result, confidence, entity, label, score)


def _entity_is_sender_signal(ent: dict, template_text: str) -> tuple[bool, float, str, str, float]:
    """
    Return (is_sender_signal, score, entity, label, score) for one aggregated NER entity.

    We only accept strong ORG entities as sender identity signals.
    Valid sender-like patterns are:
    - ORG near the end of the SMS (signature-like footer)
    - ORG near the beginning of a transactional/security/account message

    This avoids false PASS results caused by location mentions inside content,
    while still allowing app names in OTP/security-code messages.
    """
    from app.config import (
        MIN_ENTITY_LENGTH,
        ORG_SCORE_THRESHOLD,
        SENDER_ENTITY_LABELS,
        SIGNATURE_EDGE_WINDOW,
    )

    label = ent.get("entity_group", "")
    score = float(ent.get("score", 0.0))
    word = str(ent.get("word", "")).strip()

    if label not in SENDER_ENTITY_LABELS:
        return False, 0.0, "", "", 0.0
    if score < ORG_SCORE_THRESHOLD:
        return False, 0.0, "", "", 0.0
    if len(word) < MIN_ENTITY_LENGTH:
        return False, 0.0, "", "", 0.0

    start = int(ent.get("start", -1))
    end = int(ent.get("end", -1))
    text_len = len(template_text)

    near_start = start >= 0 and start <= SIGNATURE_EDGE_WINDOW
    near_end = end >= 0 and (text_len - end) <= SIGNATURE_EDGE_WINDOW
    transactional = _looks_transactional(template_text)

    if near_end:
        logger.debug(
            "Sender footer ORG entity: label=%s word=%r score=%.3f start=%s end=%s",
            label,
            word,
            score,
            start,
            end,
        )
        return True, score, word, label, score

    if near_start and transactional:
        logger.debug(
            "Sender header ORG entity in transactional SMS: label=%s word=%r score=%.3f start=%s end=%s",
            label,
            word,
            score,
            start,
            end,
        )
        return True, score, word, label, score

    logger.debug(
        "Ignoring ORG entity without sender context: label=%s word=%r score=%.3f start=%s end=%s transactional=%s",
        label,
        word,
        score,
        start,
        end,
        transactional,
    )
    return False, 0.0, "", "", 0.0


def _decide_from_entities(template_text: str, entities: list[dict]) -> ClassifyResult:
    best_score = 0.0
    best_entity = ""
    best_label = ""

    for ent in entities:
        is_signal, score, entity, label, _ = _entity_is_sender_signal(ent, template_text)
        if is_signal and score > best_score:
            best_score = score
            best_entity = entity
            best_label = label

    if best_score > 0:
        return "PASS", round(best_score, 4), best_entity, best_label, round(best_score, 4)

    return "FLAG", None, None, None, None


def classify(template_text: str) -> ClassifyResult:
    """
    Run NER inference on *template_text*.

    Returns:
        ("PASS", confidence) only when a strong ORG entity looks like a sender
            signature/footer or a sender header in a transactional message.
        ("FLAG", confidence) otherwise.
        ("FLAG", None) if the model is not available.
    """
    if not _model_loaded or _pipeline is None:
        # Try lazy load
        if not load_model():
            logger.warning(
                "Model unavailable – defaulting to FLAG for hash-unknown template"
            )
            return "FLAG", None

    try:
        entities = _pipeline(template_text)  # type: ignore[misc]
    except Exception as exc:
        logger.error("NER inference failed: %s", exc)
        return "FLAG", None

    return _decide_from_entities(template_text, entities)


def classify_batch(templates: list[str]) -> list[ClassifyResult]:
    """
    Classify a batch of template strings.
    Falls back to per-item classification if batch inference fails.
    """
    if not templates:
        return []

    if not _model_loaded:
        load_model()

    results: list[ClassifyResult] = []

    if not _model_loaded or _pipeline is None:
        return [("FLAG", None)] * len(templates)

    try:
        batch_entities = _pipeline(templates)  # type: ignore[misc]
        for template_text, ent_list in zip(templates, batch_entities):
            results.append(_decide_from_entities(template_text, ent_list))
    except Exception as exc:
        logger.error("Batch NER inference failed: %s – falling back to per-item", exc)
        results = [classify(t) for t in templates]

    return results
