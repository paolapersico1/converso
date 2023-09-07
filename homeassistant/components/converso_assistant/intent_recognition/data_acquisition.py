"""Module to load the dataset."""
import os
from os import path

import pandas as pd

from .const import DATASETS_DIR, SLOTS, USE_SAVED_GRAMMAR
from .data_preprocessing import preprocess_text, remove_stopwords
from .grammar import generate_artificial_dataset


# function to load the dataset
def load_synthetic_dataset(dataset_file_path="synthetic_dataset.csv"):
    """Load the dataset with sample commands."""
    filename = None
    if dataset_file_path:
        filename = path.join(DATASETS_DIR, dataset_file_path)
    if USE_SAVED_GRAMMAR and filename and os.access(filename, os.R_OK):
        df = pd.read_csv(
            filename,
            index_col=False,
            header=0,
            usecols=["Text", "Intent", "Domain", "DeviceClass", "State", "Response"],
            dtype={
                "Text": "str",
                "Intent": "str",
                "Domain": "str",
                "Name": "str",
                "Area": "str",
                "DeviceClass": "str",
                "Response": "str",
                "State": "str",
                "Color": "str",
                "Temperature": "float32",
                "Brightness": "float32",
            },
        )
    elif USE_SAVED_GRAMMAR:
        li = []
        for filename in [
            path.join(DATASETS_DIR, intent + ".csv") for intent in list(SLOTS.keys())
        ]:
            if os.access(filename, os.R_OK):
                df = pd.read_csv(
                    filename,
                    index_col=0,
                    header=0,
                    dtype={
                        "Text": "str",
                        "Intent": "str",
                        "Domain": "str",
                        "Name": "str",
                        "Area": "str",
                        "DeviceClass": "str",
                        "Response": "str",
                        "State": "str",
                        "Color": "str",
                        "Temperature": "float32",
                        "Brightness": "float32",
                    },
                )
                li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        df_without_stopwords = df.copy()
        tokenized_texts = df_without_stopwords["Text"].apply(
            lambda line: preprocess_text(str(line))
        )
        df_without_stopwords["Text"] = remove_stopwords(tokenized_texts).apply(" ".join)
        df = (
            df.append(df_without_stopwords, ignore_index=True)
            .drop_duplicates(subset="Text", ignore_index=True)
            .reset_index(drop=True)
        )
        df.to_csv(path.join(DATASETS_DIR, "synthetic_dataset.csv"))
    else:
        df = generate_artificial_dataset(dataset_file_path)

    return df
