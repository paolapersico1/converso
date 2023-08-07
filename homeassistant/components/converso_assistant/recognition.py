"""Module to perform the Intent Recognition task."""
from dataclasses import dataclass, field
from typing import Any, Optional

from hassil.expression import (
    TextChunk,
)
from hassil.intents import (
    Intents,
    TextSlotList,
)

from homeassistant.core import HomeAssistant


@dataclass
class MatchEntity:
    """Named entity that has been matched from a {slot_list}."""

    name: str
    """Name of the entity."""

    value: Any
    """Value of the entity."""


@dataclass
class LightRecognizeResult:
    """Result of recognition."""

    intent_name: str
    """Matched intent"""

    entities: dict[str, MatchEntity] = field(default_factory=dict)
    """Matched entities mapped by name."""

    entities_list: list[MatchEntity] = field(default_factory=list)
    """Matched entities as a list (duplicates allowed)."""

    response: Optional[str] = None
    """Key for intent response."""


def smart_intent_recognition(text: str):
    """Perform the Intent Recognition with AI."""
    intent_name = "HassGetState"
    domain = "light"
    area = "all"
    state = "none"
    response = "one_yesno"
    device_class = None

    return intent_name, domain, area, device_class, state, response


def recognize_slot(
    slot_name: str,
    input_text: str,
    slot_lists: Optional[dict[str, TextSlotList]] = None,
):
    """Recognize the slot values."""
    if slot_lists is None:
        return None
    slot_values = [
        text_slot_value.text_in
        for text_slot_value in slot_lists.get(slot_name, TextSlotList(values=[])).values
    ]

    for item in slot_values:
        if isinstance(item, TextChunk):
            chunk: TextChunk = item
        if chunk.text.lower() in input_text:
            return item
    return None


def smart_recognize_all(
    text: str,
    intents: Intents,
    hass: HomeAssistant,
    slot_lists: Optional[dict[str, TextSlotList]] = None,
):
    """Recognize the intent and fills the slots."""
    intent_name, domain, area, device_class, state, response = smart_intent_recognition(
        text
    )

    maybe_matched_entities: list[MatchEntity] = []

    entity_name = recognize_slot("name", text, slot_lists)
    if entity_name is None:
        entity_name = "all"

    maybe_matched_entities.append(MatchEntity(name="name", value=entity_name))

    if area != "none":
        area = recognize_slot("area", text, slot_lists)
        if area:
            maybe_matched_entities.append(MatchEntity(name="area", value=area))

    if domain != "none":
        maybe_matched_entities.append(MatchEntity(name="domain", value=domain))
    if device_class != "none":
        maybe_matched_entities.append(
            MatchEntity(name="device_class", value=device_class)
        )
    if state != "none":
        maybe_matched_entities.append(MatchEntity(name="state", value=state))

    return LightRecognizeResult(
        intent_name=intent_name,
        entities={entity.name: entity for entity in maybe_matched_entities},
        entities_list=maybe_matched_entities,
        response=response,
    )
