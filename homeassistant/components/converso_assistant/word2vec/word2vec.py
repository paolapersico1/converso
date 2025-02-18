"""Module to train the word2vec model."""
import logging
from os import path
from pathlib import Path

from gensim.models import KeyedVectors
import numpy as np
from numpy.linalg import norm

W2V_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)


class Word2Vec:
    """Word2Vec model."""

    def __init__(self, dim=300) -> None:
        """Initialize the model."""
        self.w2v_dim = dim
        self.w2v_model = None
        _LOGGER.debug("Fetching word2vec model")
        word2vec_model_file = (
            path.join(W2V_DIR, "gensim_cc.it." + str(self.w2v_dim)) + ".bin"
        )
        self.w2v_model = KeyedVectors.load_word2vec_format(
            word2vec_model_file, binary=True
        )
        _LOGGER.debug("Fetched word2vec model")

        self.vocab = list(self.w2v_model.index_to_key)
        self.cached_embeddings: dict = {}

    def word2vector(self, token):
        """Return the word vector for a word."""
        return self.w2v_model[token]

    def w2v_representation(self, list1):
        """Return the word vector for a list of tokens."""
        embedding = None
        if tuple(list1) in self.cached_embeddings:
            embedding = self.cached_embeddings[tuple(list1)]
        else:
            embedding = np.mean(
                [self.word2vector(token) for token in list1 if token in self.w2v_model],
                axis=0,
            )
            self.cached_embeddings[tuple(list1)] = embedding
        return embedding

    def cosine_similarity(self, list1, list2):
        """Return the cosine similarity between two text sequences."""
        result = 0
        if any(t in self.w2v_model for t in list1) and any(
            t in self.w2v_model for t in list2
        ):
            A = self.w2v_representation(list1)
            B = self.w2v_representation(list2)
            result = np.dot(A, B) / (norm(A) * norm(B))

        return result

    def get_similar_words(self, word, how_many=5):
        """Return most similar words."""
        similar_list = self.w2v_model.most_similar(
            positive=[word], topn=how_many, restrict_vocab=50000
        )
        return {item[0].lower() for item in similar_list}
