"""Intents for the light integration."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import voluptuous as vol

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, config_validation as cv, intent

from .const import DOMAIN, SERVICE_SET_TEMPERATURE

_LOGGER = logging.getLogger(__name__)

INTENT_GET = "HassClimateGetTemperature"
INTENT_SET = "HassClimateSetTemperature"


async def async_setup_intents(hass: HomeAssistant) -> None:
    """Set up the climate intents."""
    intent.async_register(hass, GetIntentHandler())
    intent.async_register(hass, SetIntentHandler())


class GetIntentHandler(intent.IntentHandler):
    """Respond with temperature."""

    intent_type = INTENT_GET
    slot_schema = {
        vol.Any("area"): cv.string,
    }

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle the hass intent."""
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)

        area_name = slots.get("area", {}).get("value")
        area: ar.AreaEntry | None = None
        if area_name is not None:
            areas = ar.async_get(hass)
            area = areas.async_get_area(area_name) or areas.async_get_area_by_name(
                area_name
            )
            if area is None:
                raise intent.IntentHandleError(f"No area named {area_name}")

        # Optional domain/device class filters.
        # Convert to sets for speed.
        domains: set[str] | None = {
            "climate",
        }
        device_classes: set[str] | None = None

        if "device_class" in slots:
            device_classes = set(slots["device_class"]["value"])

        states = list(
            intent.async_match_states(
                hass,
                area=area,
                domains=domains,
                device_classes=device_classes,
                assistant=intent_obj.assistant,
            )
        )

        if not states:
            raise intent.IntentHandleError

        _LOGGER.debug(
            "Found %s state(s) that matched: name=%s, area=%s, domains=%s, device_classes=%s, assistant=%s",
            len(states),
            area,
            domains,
            device_classes,
            intent_obj.assistant,
        )

        matched_states: list[State] = []
        unmatched_states: list[State] = []

        # Create response
        response = intent_obj.create_response()
        response.response_type = intent.IntentResponseType.QUERY_ANSWER

        success_results: list[intent.IntentResponseTarget] = []
        if area is not None:
            success_results.append(
                intent.IntentResponseTarget(
                    type=intent.IntentResponseTargetType.AREA,
                    name=area.name,
                    id=area.id,
                )
            )

        for state in states:
            success_results.append(
                intent.IntentResponseTarget(
                    type=intent.IntentResponseTargetType.ENTITY,
                    name=state.name,
                    id=state.entity_id,
                ),
            )
            matched_states.append(state)

        response.async_set_results(success_results=success_results)
        response.async_set_states(matched_states, unmatched_states)

        return response


class SetIntentHandler(intent.IntentHandler):
    """Handle set temperature intents."""

    intent_type = INTENT_SET
    slot_schema = {
        vol.Any("name", "area"): cv.string,
        vol.Optional("temperature"): vol.All(vol.Coerce(float), vol.Range(0, 100)),
    }

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle the hass intent."""
        hass = intent_obj.hass
        service_data: dict[str, Any] = {}
        slots = self.async_validate_slots(intent_obj.slots)

        name: str | None = slots.get("name", {}).get("value")
        if name == "all":
            # Don't match on name if targeting all entities
            name = None

        # Look up area first to fail early
        area_name = slots.get("area", {}).get("value")
        area: ar.AreaEntry | None = None
        if area_name is not None:
            areas = ar.async_get(hass)
            area = areas.async_get_area(area_name) or areas.async_get_area_by_name(
                area_name
            )
            if area is None:
                raise intent.IntentHandleError(f"No area named {area_name}")

        states = list(
            intent.async_match_states(
                hass,
                name=name,
                area=area,
                domains=("climate"),
            )
        )

        if not states:
            raise intent.IntentHandleError("No entities matched")

        service_data["temperature"] = slots["temperature"]["value"]

        response = intent_obj.create_response()

        success_results: list[intent.IntentResponseTarget] = []
        service_coros = []

        if area is not None:
            success_results.append(
                intent.IntentResponseTarget(
                    type=intent.IntentResponseTargetType.AREA,
                    name=area.name,
                    id=area.id,
                )
            )

        for state in states:
            target = intent.IntentResponseTarget(
                type=intent.IntentResponseTargetType.ENTITY,
                name=state.name,
                id=state.entity_id,
            )

            service_coros.append(
                hass.services.async_call(
                    DOMAIN,
                    SERVICE_SET_TEMPERATURE,
                    {**service_data, ATTR_ENTITY_ID: state.entity_id},
                    context=intent_obj.context,
                )
            )
            success_results.append(target)

        # Handle service calls in parallel.
        await asyncio.gather(*service_coros)

        response.async_set_results(success_results=success_results)

        return response
