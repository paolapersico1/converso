"""Module to preprocess the dataset."""
import logging
import os
from os import path
import re

from nltk import WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .const import DATASETS_DIR, NUMBER_DICT, SLOTS, USE_SAVED_DATASETS

_LOGGER = logging.getLogger(__name__)


class W2VTransformer(BaseEstimator, TransformerMixin):
    """W2V Transformer for Pipeline."""

    def __init__(self, w2v, with_sw=True) -> None:
        """Init for transformer."""
        self.w2v = w2v
        self.with_sw = with_sw

    def fit(self, X, y=None):
        """Fit for transformer."""
        return self

    def transform(self, X):
        """Transform for transformer."""
        X = preprocess_dataset(X, self.w2v, self.with_sw)
        return X.to_numpy()


def preprocess_dataset(arr, w2v, with_sw):
    """Create word2vec dataset."""
    df = pd.DataFrame(arr, columns=["Text"])
    df["Text"] = df["Text"].apply(lambda line: preprocess_text(str(line)))

    if not with_sw:
        df["Text"] = remove_stopwords(df["Text"])

    df["Text"] = df["Text"].astype("string")
    df = get_word2vec_dataset(df, w2v)

    return df


def create_label_datasets(label, input_df=None):
    """Create dataset for specific label."""
    filepath = path.join(
        DATASETS_DIR,
        label + "_dataset.csv",
    )
    if USE_SAVED_DATASETS and os.access(filepath, os.R_OK):
        _LOGGER.info("Fetching saved dataset for label " + label)
        df = pd.read_csv(filepath, index_col=0)
    else:
        df = input_df.copy()
        if label != "Intent":
            indices_to_drop = []
            for index, row in df.iterrows():
                if (
                    (label not in SLOTS[row["Intent"]])
                    or (label == "State" and row["Response"] == "one")
                    or (label == "DeviceClass" and row["Domain"] != "cover")
                ):
                    indices_to_drop.append(index)
            df.drop(indices_to_drop, inplace=True)

        df.drop(
            columns=[
                col
                for col in df
                if (
                    col != "Text"
                    and col != label
                    and not (label.startswith("Response") and col == "Response")
                )
            ],
            inplace=True,
        )

        df.to_csv(filepath)

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
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub(r"([0-9])\.([0-9])", r"\1FRACDOT\2", text)
    text = text.replace(".", " .")

    while "DOTDOTMULTI" in text:
        text = text.replace("DOTDOTMULTI", "DOTMULTI.")

    text = text.replace("DOTMULTI", " .")
    text = text.replace("FRACDOT", ".")

    tk = WhitespaceTokenizer()

    sequence = [str(NUMBER_DICT.get(token, token)) for token in tk.tokenize(text)]

    return sequence


def remove_stopwords(series):
    """Remove stopwords."""
    italian_stopwords = stopwords.words("italian")

    series_without_sw = series.apply(
        lambda line: [token for token in line if token not in italian_stopwords]
    )
    return series_without_sw


def mean_embedding(w2v, token_list):
    """Return the average embedding over list of tokens."""
    result = np.mean(
        [
            w2v.word2vector(token.strip(" '"))
            for token in token_list
            if token.strip(" '") in w2v.w2v_model
        ],
        axis=0,
    )
    return result


def get_word2vec_dataset(df, w2v):
    """Return a word2vec dataset."""
    df["Text"] = df["Text"].apply(
        lambda line: mean_embedding(w2v, line.strip("][").split(",")).tolist()
    )
    add_columns = pd.DataFrame(df["Text"].tolist(), index=df.index)
    df = pd.concat([df, add_columns], axis=1)
    df = df.drop(columns=["Text"])
    return df
