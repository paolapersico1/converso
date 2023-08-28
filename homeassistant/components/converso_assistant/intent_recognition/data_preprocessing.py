"""Module to preprocess the dataset."""
import logging
import os
from os import path
import re

from nltk import WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from word2vec.word2vec_training import W2V_DIM, get_word2vec_model, w2v

from .const import DATASETS_DIR, SAMPLING, SLOTS, USE_SAVED_DATASETS

_LOGGER = logging.getLogger(__name__)


def preprocess_dataset(df):
    """Create word2vec datasets (with and without stopwords)."""
    filepath1 = path.join(DATASETS_DIR, "dataset_without_sw_removal.csv")
    filepath2 = path.join(DATASETS_DIR, "dataset_with_sw_removal.csv")
    if (
        USE_SAVED_DATASETS
        and os.access(filepath1, os.R_OK)
        and os.access(filepath2, os.R_OK)
    ):
        _LOGGER.info("Fetching saved word2vec dataset")
        # print("Fetching saved word2vec dataset")
        new_df = pd.read_csv(filepath1, index_col=0)
        new_df_without_sw = pd.read_csv(filepath2, index_col=0)
    else:
        new_df = df.copy()
        new_df["Text"] = df["Text"].apply(lambda line: preprocess_text(str(line)))

        new_df_without_sw = new_df.copy()

        new_df_without_sw["Text"] = remove_stopwords(new_df_without_sw["Text"])

        new_df["Text"] = new_df["Text"].astype("string")
        new_df_without_sw["Text"] = new_df_without_sw["Text"].astype("string")

        new_df_without_sw.drop_duplicates(inplace=True)
        new_df = get_word2vec_dataset(new_df, filepath1)
        new_df_without_sw = get_word2vec_dataset(new_df_without_sw, filepath2)

    subsets = {"without_sw_removal": new_df, "with_sw_removal": new_df_without_sw}

    return subsets


def create_label_datasets(df, df_without_sw, label):
    """Create dataset for specific label."""
    filepath1 = path.join(
        DATASETS_DIR, label + "_dataset_without_sw_removal_" + SAMPLING + ".csv"
    )
    filepath2 = path.join(
        DATASETS_DIR, label + "_dataset_with_sw_removal_" + SAMPLING + ".csv"
    )
    filepath3 = path.join(DATASETS_DIR, label + "_full_dataset_" + SAMPLING + ".csv")
    if (
        USE_SAVED_DATASETS
        and os.access(filepath1, os.R_OK)
        and os.access(filepath2, os.R_OK)
        and os.access(filepath3, os.R_OK)
    ):
        _LOGGER.info("Fetching saved dataset for label " + label)
        # print("Fetching saved dataset for label " + label)
        df = pd.read_csv(filepath1, index_col=0)
        df_without_sw = pd.read_csv(filepath2, index_col=0)
        full_df = pd.read_csv(filepath3, index_col=0)
    else:
        for dataframe in (df, df_without_sw):
            if label != "Intent":
                indices_to_drop = []
                for index, row in dataframe.iterrows():
                    if (
                        (label not in SLOTS[row["Intent"]])
                        or (label == "State" and row["Response"] == "one")
                        or (label == "DeviceClass" and row["Domain"] != "cover")
                    ):
                        indices_to_drop.append(index)
                dataframe.drop(indices_to_drop, inplace=True)

            dataframe.drop(
                columns=[
                    col
                    for col in dataframe
                    if (
                        (
                            col != label
                            and not (label.startswith("Response") and col == "Response")
                        )
                        and (not col.startswith("v_"))
                    )
                ],
                inplace=True,
            )

        df.to_csv(filepath1)
        df_without_sw.to_csv(filepath2)
        full_df = pd.concat([df, df_without_sw], ignore_index=True)
        full_df.drop_duplicates(inplace=True)
        full_df.to_csv(filepath3)

    subsets = {
        "without_sw_removal_" + SAMPLING: df,
        "with_sw_removal_" + SAMPLING: df_without_sw,
        "full_" + SAMPLING: full_df,
    }

    return subsets


def preprocess_text(text):
    """Tokenize the text."""
    text = text.lower()
    text = text.strip()
    text = " " + text + " "
    text = re.sub(r"([^\w\s.\'\`,\-])", r" \1 ", text)

    text = re.sub(r"\.([\.]+)", r"DOTMULTI\1", text)
    while "DOTMULTI." in text:
        text = re.sub(r"DOTMULTI\.([^\.])", r"DOTDOTMULTI \1", text)
        text = re.sub(r"DOTMULTI\.", r"DOTDOTMULTI", text)

    text = re.sub(r"([^0-9]),([^0-9])", r"\1 , \2", text)
    text = re.sub(r"([0-9]),([^0-9])", r"\1 , \2", text)
    text = re.sub(r"([^0-9]),([0-9])", r"\1 , \2", text)
    text = text.replace("`", "'")
    text = text.replace("''", ' " ')
    text = re.sub(r"([^\w])[']([^\w])", r"\1 ' \2", text)
    text = re.sub(r"([^\w])[']([\w])", r"\1 ' \2", text)
    text = re.sub(r"([\w])[']([^\w])", r"\1 ' \2", text)
    text = re.sub(r"([\w])[']([\w])", r"\1' \2", text)
    text = text.lstrip()
    text = text.rstrip()
    text = text.replace(".", " .")

    while "DOTDOTMULTI" in text:
        text = text.replace("DOTDOTMULTI", "DOTMULTI.")

    text = text.replace("DOTMULTI", " .")

    tk = WhitespaceTokenizer()

    return tk.tokenize(text)


def remove_stopwords(series):
    """Remove stopwords."""
    italian_stopwords = stopwords.words("italian")

    series_without_sw = series.apply(
        lambda line: [token for token in line if token not in italian_stopwords]
    )
    return series_without_sw


def full_text_preprocess(w2v_model, text):
    """Return a dataframe with a vector representation of a text and the list of tokens."""
    text = preprocess_text(text)
    df = pd.DataFrame(columns=["v_" + str(i) for i in range(W2V_DIM)])
    df.loc[0] = np.mean([w2v(w2v_model, token.strip(" '")) for token in text], axis=0)
    return df, text


def get_word2vec_dataset(df, filepath):
    """Return a word2vec dataset."""
    _LOGGER.info("Creating a new word2vec dataset")
    # print("Creating a new word2vec dataset")
    w2v_model = get_word2vec_model()
    vectors = df["Text"].apply(
        lambda line: np.mean(
            [
                w2v(w2v_model, token.strip(" '"))
                for token in line.strip("][").split(",")
            ],
            axis=0,
        ).tolist()
    )
    text_cols = ["v_" + str(i) for i in range(W2V_DIM)]
    df[text_cols] = list(vectors.values)
    # df = df.drop(columns=["Text"])
    df.to_csv(filepath)

    _LOGGER.info("The " + filepath + " dataset is ready to be used")
    # print("The " + filepath + " dataset is ready to be used")
    return df
