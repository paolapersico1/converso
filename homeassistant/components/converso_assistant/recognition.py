"""Module to perform the Intent Recognition task."""
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Optional

from hassil.expression import TextChunk
from hassil.intents import Intents, TextSlotList
from nltk.util import ngrams
import numpy as np

from homeassistant.core import HomeAssistant

from .intent_recognition.classification import load_and_predict
from .intent_recognition.const import COLORS
from .intent_recognition.data_preprocessing import preprocess_text

_LOGGER = logging.getLogger(__name__)

PUNCTUATION = re.compile(r"[.。,，?¿？؟!！;；:：]+")


@dataclass
class MatchEntity:
    """Named entity that has been matched from a {slot_list}."""

    name: str
    """Name of the entity."""

    value: Any
    """Value of the entity."""

    text: str
    """Original value text."""


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


class IntentRecognizer:
    """Intent recognition for text commands."""

    def __init__(self, w2v) -> None:
        """Initialize the engine."""
        self.w2v = w2v
        self.tokens: list[list[Any]] = []
        self.ngrams: list[list[Any]] = []

    def extract_value(self, valid_values, threshold: float = 0.95):
        """Extract value from list of tokens."""
        max_similarity = -1.0
        best = None
        for item in valid_values:
            for ngram in self.ngrams:
                sim = self.w2v.cosine_similarity(preprocess_text(str(item)), ngram)
                if sim >= threshold and sim > max_similarity:
                    max_similarity = sim
                    best = item
        return best

    def recognize_slot(
        self,
        slot_name: str,
        slot_lists: Optional[dict[str, TextSlotList]] = None,
        threshold: float = 0.95,
    ) -> list:
        """Recognize the slot values."""
        if slot_lists is None:
            return []
        slot_values = [
            text_slot_value.text_in
            for text_slot_value in slot_lists.get(
                slot_name, TextSlotList(values=[])
            ).values
        ]

        results = []
        for item in slot_values:
            if isinstance(item, TextChunk):
                chunk: TextChunk = item
                for ngram in self.ngrams:
                    if (
                        self.w2v.cosine_similarity(preprocess_text(chunk.text), ngram)
                        >= threshold
                    ):
                        results.append(chunk.text)
        return results

    def smart_recognize_all(
        self,
        text: str,
        intents: Intents,
        hass: HomeAssistant,
        slot_lists: Optional[dict[str, TextSlotList]] = None,
    ) -> list[LightRecognizeResult]:
        """Recognize the intent and fills the slots."""
        tokens = preprocess_text(text)
        self.tokens = [[token] for token in tokens]
        bigrams = [list(ngram) for ngram in list(ngrams(tokens, 2))]
        trigrams = [list(ngram) for ngram in list(ngrams(tokens, 3))]
        self.ngrams = self.tokens + bigrams + trigrams
        result = load_and_predict(text, "svc_linear__full_undersampling")
        _LOGGER.debug(result)

        maybe_matched_entities: list[MatchEntity] = []

        names = self.recognize_slot("name", slot_lists)
        if not names:
            names = ["all"]

        areas = self.recognize_slot("area", slot_lists)

        if result.get("Domain", "default") != "default":
            maybe_matched_entities.append(
                MatchEntity(
                    name="domain", value=result["Domain"], text=result["Domain"]
                )
            )
        if result.get("DeviceClass", "none") != "none":
            maybe_matched_entities.append(
                MatchEntity(
                    name="device_class",
                    value=result["DeviceClass"],
                    text=result["DeviceClass"],
                )
            )
        if result.get("State", "none") != "none":
            maybe_matched_entities.append(
                MatchEntity(name="state", value=result["State"], text=result["State"])
            )

        response = result.get("Response", "default")

        if response.startswith("brightness"):
            brightness = self.extract_value(np.arange(0, 101, 1))
            if brightness is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="brightness",
                        value=brightness,
                        text=brightness,
                    )
                )
        if response.startswith("color"):
            color = self.extract_value(COLORS.keys())
            if color is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="color",
                        value=COLORS[color.capitalize()],
                        text=color.lower(),
                    )
                )

        if result["Intent"] == "HassClimateSetTemperature":
            temperature = self.extract_value(np.arange(10, 30, 0.1))
            if temperature is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="temperature",
                        value=temperature,
                        text=temperature,
                    )
                )

        if areas:
            results = [
                LightRecognizeResult(
                    intent_name=result["Intent"],
                    entities={
                        entity.name: entity
                        for entity in maybe_matched_entities
                        + [
                            MatchEntity(name="name", value=name, text=name),
                            MatchEntity(name="area", value=area, text=area),
                        ]
                    },
                    entities_list=maybe_matched_entities
                    + [
                        MatchEntity(name="name", value=name, text=name),
                        MatchEntity(name="area", value=area, text=area),
                    ],
                    response=response,
                )
                for name in names
                for area in areas
            ]
        else:
            results = [
                LightRecognizeResult(
                    intent_name=result["Intent"],
                    entities={
                        entity.name: entity
                        for entity in maybe_matched_entities
                        + [MatchEntity(name="name", value=name, text=name)]
                    },
                    entities_list=maybe_matched_entities
                    + [MatchEntity(name="name", value=name, text=name)],
                    response=response,
                )
                for name in names
            ]

        return results
