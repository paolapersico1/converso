"""Const for conversation integration."""

from pathlib import Path

DOMAIN = "conversation"
DEFAULT_EXPOSED_ATTRIBUTES = {"device_class"}
HOME_ASSISTANT_AGENT = "homeassistant"
CONVERSATION_ROOT = Path(__file__).parent
