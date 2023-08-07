"""Module to preprocess the dataset."""
import logging
import os
from os import path

import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

from .consts import (
    DATASETS_DIR,
    USE_SAVED_DATASETS,
)
from .word2vec.word2vec_training import get_word2vec_model, w2v

_LOGGER = logging.getLogger(__name__)


def create_datasets(df):
    """Create word2vec datasets (with and without stopwords)."""
    new_df = df.copy()
    new_df_without_sw = df.copy()

    new_df["Text"] = df["Text"].apply(lambda line: preprocess_text(str(line)))
    new_df["Text"] = new_df["Text"].astype("string")

    new_df_without_sw["Text"] = remove_stopwords(new_df["Text"])
    new_df_without_sw["Text"] = new_df_without_sw["Text"].astype("string")
    new_df_without_sw.drop_duplicates()

    new_df = balance_dataset(new_df)
    new_df_without_sw = balance_dataset(new_df_without_sw)

    x_subsets = {
        "without_sw_removal": get_word2vec_dataset(
            new_df,
            path.join(DATASETS_DIR, "without_sw_removal.csv"),
        ),
        "with_sw_removal": get_word2vec_dataset(
            new_df_without_sw,
            path.join(DATASETS_DIR, "with_sw_removal.csv"),
        ),
    }

    return x_subsets


def balance_dataset(df):
    """Balance dataset in respect to Intent."""
    g = df.groupby("Intent")
    df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    return df


def preprocess_text(line):
    """Tokenize the text."""
    # nlp = it_core_news_lg.load()
    # df['Text'] = [[w.lemma_ for w in nlp(line)] for line in df[column_name]]

    return nltk.word_tokenize(line, language="italian")


def remove_stopwords(series):
    """Remove stopwords."""
    nltk.download("stopwords")
    italian_stopwords = stopwords.words("italian")

    series_without_sw = series.apply(
        lambda line: [token for token in line if token not in italian_stopwords]
    )
    return series_without_sw


def get_word2vec_dataset(df, filepath):
    """Return a word2vec dataset."""
    if USE_SAVED_DATASETS and os.access(filepath, os.R_OK):
        _LOGGER.info("Fetching saved word2vec dataset")
        df = pd.read_csv(filepath, index_col=0)
    else:
        _LOGGER.info("Creating a new word2vec dataset")
        w2v_model = get_word2vec_model()
        # Store the vectors for data in a file
        df["Text"] = df["Text"].apply(
            lambda line: np.mean(
                [w2v(w2v_model, token) for token in line], axis=0
            ).tolist()
        )
        text_cols = ["v_" + str(i) for i in range(300)]
        df[text_cols] = list(df["Text"].values)
        df = df.drop(columns=["Text"])
        df.to_csv(filepath)

    _LOGGER.info("The " + filepath + " dataset is ready to be used")
    return df
