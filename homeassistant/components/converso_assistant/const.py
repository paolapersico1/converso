"""Constants for the Converso Assistant integration."""
from pathlib import Path

DOMAIN = "converso_assistant"
DEFAULT_EXPOSED_ATTRIBUTES = {"device_class"}
CONVERSO_ROOT = Path(__file__).parent
NGRAMS_DIR = CONVERSO_ROOT / "ngrams"
