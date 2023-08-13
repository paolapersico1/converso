"""Constants."""
from pathlib import Path

ROOT = Path(__file__).parent

MODELS_DIR = ROOT / "trained_models"
DATASETS_DIR = ROOT / "datasets"
USE_SAVED_GRAMMAR = True
USE_SAVED_DATASETS = True
USE_SAVED_MODELS = True
SAVE_MODELS = True
SLOTS = {
    "HassTurnOn": ("Domain", "DeviceClass"),
    "HassTurnOff": ("Domain", "DeviceClass"),
    "HassGetState": ("Domain", "DeviceClass", "State", "Response"),
    "HassLightSet": ("Response",),
    "HassClimateGetTemperature": (),
    "HassClimateSetTemperature": (),
}

COLORS = [
    "Giallo",
    "Bianco",
    "Viola",
    "Rosso",
    "Arancione",
    "Marrone",
    "Blu",
    "Verde",
    "Nero",
]
