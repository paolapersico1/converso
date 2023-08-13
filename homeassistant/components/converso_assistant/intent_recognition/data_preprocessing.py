"""Module to preprocess the dataset."""
import logging
import os
from os import path
import re

from nltk import WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

from .const import (
    DATASETS_DIR,
    SLOTS,
    USE_SAVED_DATASETS,
)
from .word2vec.word2vec_training import (
    W2V_DIM,
    get_word2vec_model,
    w2v,
)

_LOGGER = logging.getLogger(__name__)


def create_datasets(df, label):
    """Create word2vec datasets (with and without stopwords)."""
    new_df = df.copy()
    if label != "Intent":
        indices_to_drop = []
        for index, row in new_df.iterrows():
            if (
                (label not in SLOTS[row["Intent"]])
                or (label == "State" and row["Response"] == "one")
                or (label == "Brightness" and row["Response"] != "brightness")
                or (label == "Color" and row["Response"] != "color")
                or (label == "DeviceClass" and row["Domain"] != "cover")
            ):
                indices_to_drop.append(index)
        new_df.drop(indices_to_drop, inplace=True)
    new_df.drop(
        columns=[col for col in new_df if col not in (label, "Text")],
        inplace=True,
    )

    new_df["Text"] = df["Text"].apply(lambda line: preprocess_text(str(line)))

    new_df_without_sw = new_df.copy()

    new_df_without_sw["Text"] = remove_stopwords(new_df_without_sw["Text"])

    new_df["Text"] = new_df["Text"].astype("string")
    new_df_without_sw["Text"] = new_df_without_sw["Text"].astype("string")

    new_df_without_sw.drop_duplicates(inplace=True)
    if label in ("Brightness", "Temperature"):
        new_df[label] = new_df[label].astype("float")
        new_df_without_sw[label] = new_df_without_sw[label].astype("float")

    new_df = balance_dataset(new_df, label)
    new_df_without_sw = balance_dataset(new_df_without_sw, label)

    x_subsets = {
        "without_sw_removal": get_word2vec_dataset(
            new_df,
            path.join(DATASETS_DIR, label + "_without_sw_removal.csv"),
        ),
        "with_sw_removal": get_word2vec_dataset(
            new_df_without_sw,
            path.join(DATASETS_DIR, label + "_with_sw_removal.csv"),
        ),
    }

    return x_subsets


def balance_dataset(df, label):
    """Balance dataset in respect to label."""
    g = df.groupby(label)
    df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    return df


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
    text = text.replace(" +", " ")
    text = text.lstrip()
    text = text.rstrip()

    while "DOTDOTMULTI" in text:
        text = text.replace("DOTDOTMULTI", "DOTMULTI.")

    text = text.replace("DOTMULTI", ".")

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
    """Return a vector representation of a text."""
    text = preprocess_text(text)
    df = pd.DataFrame(columns=["v_" + str(i) for i in range(W2V_DIM)])
    df.loc[0] = np.mean([w2v(w2v_model, token.strip(" '")) for token in text], axis=0)
    return df


def get_word2vec_dataset(df, filepath):
    """Return a word2vec dataset."""
    if USE_SAVED_DATASETS and os.access(filepath, os.R_OK):
        _LOGGER.info("Fetching saved word2vec dataset")
        # print("Fetching saved word2vec dataset")
        df = pd.read_csv(filepath, index_col=0)
    else:
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
    return df
