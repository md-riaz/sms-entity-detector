"""
Manual helper to preload the HuggingFace NER model into the local cache.

Usage:
  python -m app.preload_model
"""

from app import classifier
from app.config import NER_MODEL_NAME, MODEL_ENABLED


def main() -> int:
    if not MODEL_ENABLED:
        print("MODEL_ENABLED=false — skipping preload")
        return 0

    print(f"Preloading model: {NER_MODEL_NAME}")
    ok = classifier.load_model()
    if ok:
        print("Model preloaded successfully")
        return 0

    error = classifier.model_error() or "unknown error"
    print(f"Model preload failed: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
