"""Constants."""
from pathlib import Path

ROOT = Path(__file__).parent

MODELS_DIR = ROOT / "trained_models"
DATASETS_DIR = ROOT / "datasets"
USE_SAVED_GRAMMAR = True
USE_SAVED_DATASETS = True
USE_SAVED_MODELS = True
SAVE_MODELS = True
WORD2VEC_DIM = 100

SAMPLING = "oversampling"
STOP_WORD_REMOVAL = False
SLOTS = {
    "HassTurnOn": ("Domain", "DeviceClass", "ResponseHassTurn"),
    "HassTurnOff": ("Domain", "DeviceClass", "ResponseHassTurn"),
    "HassGetState": ("Domain", "DeviceClass", "State", "ResponseHassGetState"),
    "HassLightSet": ("ResponseHassLightSet",),
    "HassClimateGetTemperature": (),
    "HassClimateSetTemperature": (),
}

COLORS = {
    "Giallo": "yellow",
    "Bianco": "white",
    "Viola": "purple",
    "Rosso": "red",
    "Arancione": "orange",
    "Marrone": "brown",
    "Blu": "blue",
    "Verde": "green",
    "Nero": "black",
}
