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
    "HassTurnOn": ("Name", "Domain", "Area", "DeviceClass"),
    "HassTurnOff": ("Name", "Domain", "Area", "DeviceClass"),
    "HassGetState": ("Name", "Domain", "Area", "DeviceClass", "State", "Response"),
    "HassLightSet": ("Name", "Area", "Response"),
    "HassClimateGetTemperature": ("Area",),
    "HassClimateSetTemperature": ("Area",),
}
