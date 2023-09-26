"""Module to perform the Intent Recognition task."""
from dataclasses import dataclass, field
import logging
from typing import Any, Optional

from hassil.expression import TextChunk
from hassil.intents import Intents, TextSlotList
from nltk.util import ngrams
import spacy

from homeassistant.core import HomeAssistant

from .intent_recognition.classification import load_and_predict
from .intent_recognition.const import COLORS
from .intent_recognition.data_preprocessing import preprocess_text

_LOGGER = logging.getLogger(__name__)


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
class SmartRecognizeResult:
    """Result of recognition."""

    intent_name: str
    """Matched intent"""

    entities: dict[str, MatchEntity] = field(default_factory=dict)
    """Matched entities mapped by name."""

    entities_list: list[MatchEntity] = field(default_factory=list)
    """Matched entities as a list (duplicates allowed)."""

    response: Optional[str] = None
    """Key for intent response."""

    text: str = ""
    """Input text for recognition."""


class IntentRecognizer:
    """Intent recognition for text commands."""

    def __init__(self, w2v) -> None:
        """Initialize the engine."""
        self.w2v = w2v
        self.tokens: list[list[Any]] = []
        self.ngrams: list[list[Any]] = []
        self.temperatures = [n / 10.0 for n in range(0, 301)]
        self.brightness = range(0, 101)
        self.nlp = spacy.load("it_core_news_sm")

    def extract_decimal_part(self, subtree):
        """Extract decimal part from text."""
        skip = True
        for sub_tok in subtree:
            if "nummod" in sub_tok.dep_ or "conj" in sub_tok.dep_:
                if not skip:
                    if str(sub_tok) == "mezzo":
                        return 0.5
                    if str(sub_tok).isnumeric():
                        return 0.1 * float(str(sub_tok))
                skip = False
        return 0

    def extract_number(self, text, valid_values):
        """Extract number from text."""
        doc = self.nlp(text)
        result = None
        for token in doc:
            if "nummod" in token.dep_ and str(token).isnumeric():
                number = float(str(token))
                if number:
                    subtree = list(token.head.subtree)
                    number = number + self.extract_decimal_part(subtree)

                    if number in valid_values:
                        result = number
        return result

    def extract_color(self, text):
        """Extract color from text."""
        for color_it, color_en in COLORS.items():
            color_it = color_it.lower()
            if color_it in text:
                return [color_en, color_it]
        return None

    def recognize_slot(
        self,
        slot_name: str,
        slot_lists: Optional[dict[str, TextSlotList]] = None,
        threshold: float = 0.999,
    ) -> list:
        """Recognize the slot values."""
        if slot_lists is None:
            return []
        slot_values = list(slot_lists.get(slot_name, TextSlotList(values=[])).values)

        results: list = []
        for item in slot_values:
            if isinstance(item.text_in, TextChunk):
                chunk: TextChunk = item.text_in
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
    ) -> list[SmartRecognizeResult] | None:
        """Recognize the intent and fills the slots."""
        tokens = preprocess_text(text)
        preprocessed_text = " ".join(tokens)

        self.tokens = [[token] for token in tokens]
        bigrams = [list(ngram) for ngram in list(ngrams(tokens, 2))]
        trigrams = [list(ngram) for ngram in list(ngrams(tokens, 3))]
        self.ngrams = self.tokens + bigrams + trigrams

        result = load_and_predict(text, self.w2v)
        _LOGGER.debug(result)
        if not result:
            return None

        maybe_matched_entities: list[MatchEntity] = []

        domain = None
        device_class = None

        if result.get("Domain", "default") != "default":
            domain = result["Domain"]

        if result.get("DeviceClass", "none") != "none":
            device_class = result["DeviceClass"].lower()

        areas = self.recognize_slot("area", slot_lists)
        names = self.recognize_slot("name", slot_lists)
        if not names:
            names = ["all"]
            if domain:
                maybe_matched_entities.append(
                    MatchEntity(name="domain", value=domain, text=domain)
                )
            if device_class:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="device_class",
                        value=device_class,
                        text=device_class,
                    )
                )

        if result.get("State", "none") != "none":
            maybe_matched_entities.append(
                MatchEntity(name="state", value=result["State"], text=result["State"])
            )

        response = result.get("Response", "default")

        if response.startswith("brightness"):
            brightness = self.extract_number(preprocessed_text, self.brightness)
            if brightness is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="brightness",
                        value=brightness,
                        text=brightness,
                    )
                )
            else:
                return None
        if response.startswith("color"):
            color = self.extract_color(preprocessed_text)
            if color is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="color",
                        value=color[0],
                        text=color[1],
                    )
                )
            else:
                return None

        if result["Intent"] == "HassClimateSetTemperature":
            temperature = self.extract_number(preprocessed_text, self.temperatures)
            if temperature is not None:
                maybe_matched_entities.append(
                    MatchEntity(
                        name="temperature",
                        value=temperature,
                        text=temperature,
                    )
                )
            else:
                return None

        if areas:
            results = [
                SmartRecognizeResult(
                    text=text,
                    intent_name=result["Intent"],
                    entities={
                        entity.name: entity
                        for entity in self.add_name_area(
                            maybe_matched_entities, name, area
                        )
                    },
                    entities_list=self.add_name_area(
                        maybe_matched_entities, name, area
                    ),
                    response=response,
                )
                for name in names
                for area in areas
            ]
        else:
            results = [
                SmartRecognizeResult(
                    text=text,
                    intent_name=result["Intent"],
                    entities={
                        entity.name: entity
                        for entity in self.add_name_area(maybe_matched_entities, name)
                    },
                    entities_list=self.add_name_area(maybe_matched_entities, name),
                    response=response,
                )
                for name in names
            ]

        return results

    def add_name_area(self, entities, name, area=None):
        """Add name and area to matched entities."""
        result = entities
        if name != "all":
            result.append(MatchEntity(name="name", value=name, text=name))
        if area:
            result.append(MatchEntity(name="area", value=area, text=area))
        return result
