"""Module to load the dataset."""
import os
from os import path

import pandas as pd

from .const import DATASETS_DIR, USE_SAVED_GRAMMAR
from .data_preprocessing import preprocess_text, remove_stopwords
from .grammar import generate_artificial_dataset


# function to load the dataset
def load_synthetic_dataset(filename="synthetic_dataset.csv"):
    """Load the dataset with sample commands."""
    dataset_file_path = path.join(DATASETS_DIR, filename)
    grammar_file_path = path.join(DATASETS_DIR, "grammar_" + filename)
    if USE_SAVED_GRAMMAR and os.access(dataset_file_path, os.R_OK):
        df = pd.read_csv(
            dataset_file_path,
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
    elif USE_SAVED_GRAMMAR and os.access(grammar_file_path, os.R_OK):
        df = pd.read_csv(
            grammar_file_path,
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

        df_without_stopwords = df.copy()
        tokenized_texts = df_without_stopwords["Text"].apply(
            lambda line: preprocess_text(str(line))
        )
        df_without_stopwords["Text"] = remove_stopwords(tokenized_texts).apply(" ".join)
        df = (
            pd.concat([df, df_without_stopwords], ignore_index=True)
            .drop_duplicates(subset="Text", ignore_index=True)
            .reset_index(drop=True)
        )
        df.to_csv(dataset_file_path)
    else:
        df = generate_artificial_dataset(grammar_file_path)

    return df
