"""Module to load the dataset."""
import os
from os import path

import pandas as pd

from .const import (
    DATASETS_DIR,
    SLOTS,
    USE_SAVED_GRAMMAR,
)
from .grammar import (
    generate_artificial_dataset,
)


# function to load the dataset
def load_synthetic_dataset(dataset_file_path="synthetic_dataset.csv"):
    """Load the dataset with sample commands."""
    filename = None
    if dataset_file_path:
        filename = path.join(DATASETS_DIR, dataset_file_path)
    if USE_SAVED_GRAMMAR and filename and os.access(filename, os.R_OK):
        df = pd.read_csv(filename, index_col=0, header=0)
    elif USE_SAVED_GRAMMAR:
        li = []
        for filename in [
            path.join(DATASETS_DIR, intent + ".csv") for intent in list(SLOTS.keys())
        ]:
            if os.access(filename, os.R_OK):
                df = pd.read_csv(filename, index_col=0, header=0)
                li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        df.to_csv(path.join(DATASETS_DIR, "synthetic_dataset.csv"))
    else:
        df = generate_artificial_dataset(dataset_file_path)

    return df
