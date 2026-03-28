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

ClassifyResult = tuple[str, Optional[float]]  # (result, confidence)


def classify(template_text: str) -> ClassifyResult:
    """
    Run NER inference on *template_text*.

    Returns:
        ("PASS", confidence) if an ORG/PER/LOC entity with high confidence
            is detected (signals sender identity).
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

    from app.config import MIN_ENTITY_LENGTH, ORG_LABELS, ORG_SCORE_THRESHOLD

    try:
        entities = _pipeline(template_text)  # type: ignore[misc]
    except Exception as exc:
        logger.error("NER inference failed: %s", exc)
        return "FLAG", None

    best_score: float = 0.0

    for ent in entities:
        label: str = ent.get("entity_group", "")
        score: float = float(ent.get("score", 0.0))
        word: str = ent.get("word", "")

        if (
            label in ORG_LABELS
            and score >= ORG_SCORE_THRESHOLD
            and len(word.strip()) >= MIN_ENTITY_LENGTH
        ):
            if score > best_score:
                best_score = score
            logger.debug(
                "NER entity: label=%s word=%r score=%.3f", label, word, score
            )

    if best_score >= ORG_SCORE_THRESHOLD:
        return "PASS", round(best_score, 4)

    # No strong org/entity signal
    return "FLAG", round(1.0 - best_score, 4) if best_score > 0 else None


def classify_batch(templates: list[str]) -> list[ClassifyResult]:
    """
    Classify a batch of template strings.
    Falls back to per-item classification if batch inference fails.
    """
    if not templates:
        return []

    if not _model_loaded:
        load_model()

    from app.config import MIN_ENTITY_LENGTH, ORG_LABELS, ORG_SCORE_THRESHOLD

    results: list[ClassifyResult] = []

    if not _model_loaded or _pipeline is None:
        return [("FLAG", None)] * len(templates)

    try:
        batch_entities = _pipeline(templates)  # type: ignore[misc]
        for ent_list in batch_entities:
            best_score: float = 0.0
            for ent in ent_list:
                label = ent.get("entity_group", "")
                score = float(ent.get("score", 0.0))
                word = ent.get("word", "")
                if (
                    label in ORG_LABELS
                    and score >= ORG_SCORE_THRESHOLD
                    and len(word.strip()) >= MIN_ENTITY_LENGTH
                ):
                    if score > best_score:
                        best_score = score
            if best_score >= ORG_SCORE_THRESHOLD:
                results.append(("PASS", round(best_score, 4)))
            else:
                results.append(
                    ("FLAG", round(1.0 - best_score, 4) if best_score > 0 else None)
                )
    except Exception as exc:
        logger.error("Batch NER inference failed: %s – falling back to per-item", exc)
        results = [classify(t) for t in templates]

    return results
