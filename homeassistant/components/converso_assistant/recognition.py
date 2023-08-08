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

from .intent_recognition.classification import load_and_predict
from .intent_recognition.data_preprocessing import full_text_preprocess


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


def smart_intent_recognition(text: str, w2v_model):
    """Perform the Intent Recognition task with AI."""

    X = full_text_preprocess(w2v_model, text)
    result = load_and_predict(X, "svc_linear__without_sw_removal")

    return result


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
    w2v_model: Any,
    slot_lists: Optional[dict[str, TextSlotList]] = None,
):
    """Recognize the intent and fills the slots."""
    result = smart_intent_recognition(text, w2v_model)

    maybe_matched_entities: list[MatchEntity] = []

    name = "all"
    if result["Name"] != "none":
        name = recognize_slot("name", text, slot_lists)

    maybe_matched_entities.append(MatchEntity(name="name", value=name))

    if result["Area"] != "none":
        area = recognize_slot("area", text, slot_lists)
        if area:
            maybe_matched_entities.append(MatchEntity(name="area", value=area))

    if result["Domain"] != "none":
        maybe_matched_entities.append(
            MatchEntity(name="domain", value=result["Domain"])
        )
    if result["DeviceClass"] != "none":
        maybe_matched_entities.append(
            MatchEntity(name="device_class", value=result["DeviceClass"])
        )
    if result["State"] != "none":
        maybe_matched_entities.append(MatchEntity(name="state", value=result["State"]))

    return LightRecognizeResult(
        intent_name=result["Intent"],
        entities={entity.name: entity for entity in maybe_matched_entities},
        entities_list=maybe_matched_entities,
        response=result["Response"],
    )
