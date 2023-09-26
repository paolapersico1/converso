"""Constants."""
from pathlib import Path

from num2words import num2words

ROOT = Path(__file__).parent

WORD2VEC_DIM = 100
MODELS_DIR = ROOT / "new_trained_models"
DATASETS_DIR = ROOT / "new_datasets"
USE_SAVED_GRAMMAR = True
USE_SAVED_DATASETS = True
USE_SAVED_MODELS = False
SAVE_MODELS = True

SAMPLING = "undersampling"
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

NUMBER_DICT = {}
for i in range(0, 100):
    words = num2words(i, lang="it")
    NUMBER_DICT[words] = i
    if "é" in words:
        NUMBER_DICT[words.replace("é", "e")] = i
