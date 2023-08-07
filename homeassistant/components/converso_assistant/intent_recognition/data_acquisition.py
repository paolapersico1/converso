"""Module to load the dataset."""
import os
from os import path

import pandas as pd

from .consts import (
    DATASETS_DIR,
    USE_SAVED_GRAMMAR,
)
from .grammar import generate_artificial_dataset


# function to load the dataset
def load_synthetic_dataset():
    """Load the dataset with sample commands."""
    dataset_file_path = path.join(DATASETS_DIR, "synthetic_dataset.csv")

    if USE_SAVED_GRAMMAR and os.access(dataset_file_path, os.R_OK):
        df = pd.read_csv(dataset_file_path, index_col=0)
    else:
        df = generate_artificial_dataset(dataset_file_path)

    return df
