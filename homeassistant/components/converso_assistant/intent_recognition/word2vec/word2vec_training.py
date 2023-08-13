"""Module to train the word2vec model."""
import logging
import os
from os import path
from pathlib import Path

import fasttext

W2V_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)
W2V_DIM = 100


def get_word2vec_model(use_saved=True, save_models=True):
    """Load or create a word2vec model."""
    word2vec_model_file = path.join(W2V_DIR, "cc.it." + str(W2V_DIM) + ".bin")
    w2v_model = None
    if use_saved and os.access(word2vec_model_file, os.R_OK):
        _LOGGER.info("Fetching word2vec model")
        # load the model
        w2v_model = fasttext.load_model(word2vec_model_file)
        # w2v_model = Word2Vec.load(word2vec_model_file)

    return w2v_model


def w2v(w2v_model, token):
    """Return the word vector for a word."""
    return w2v_model.get_word_vector(token)


def w2v_words(w2v_model):
    """Return the vocabulary."""
    return w2v_model.words
