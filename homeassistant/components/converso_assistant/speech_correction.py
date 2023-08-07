"""Module to perform the speech correction."""
import logging

import nltk
from nltk import word_tokenize
from nltk.util import ngrams

from .intent_recognition.data_acquisition import (
    load_synthetic_dataset,
)

_LOGGER = logging.getLogger(__name__)


class SpeechCorrector:
    """Speech corrector for text commands."""

    def __init__(self, vocab) -> None:
        """Initialize the corrector."""
        self.vocab = vocab
        df = load_synthetic_dataset()
        self.prob_bi = self.train_ngram(df["Text"])

    def train_ngram(self, corpus):
        """Train the n-gram model."""
        bigram = []
        trigram = []

        for sentence in corpus:
            sentence = sentence.lower()
            sequence = word_tokenize(sentence)
            # for word in sequence:
            # if (word == '.'):
            # sequence.remove(word)

            bigram.extend(list(ngrams(sequence, 2)))
            trigram.extend(list(ngrams(sequence, 3)))

        freq_bi = nltk.FreqDist(bigram)
        # freq_tri = nltk.FreqDist(trigram)

        return nltk.MLEProbDist(freq_bi)

    def correct(self, sentence):
        """Return a corrected sentence."""
        threshold = 0.000001

        sentence = sentence.lower()
        sequence = word_tokenize(sentence)

        # compute bi-gram probabilities of sentence
        for bigram in list(ngrams(sequence, 2)):
            prob = self.prob_bi.prob(bigram)
            _LOGGER.debug(str(bigram) + " probability: " + str(prob))
            if prob < threshold:
                bigram_as_str = " ".join(bigram)
                cand1 = self.legit(self.edits1(bigram_as_str))
                if cand1:
                    sentence = sentence.replace(
                        bigram_as_str, self.choose_best_candidate(cand1)
                    )

        return sentence

    def known(self, text):
        """Return true if the word is in the vocabulary."""
        result = True
        for t in word_tokenize(text):
            if t not in self.vocab:
                result = False
        return result

    def legit(self, words):
        """Return the subset of words that appear in the dictionary."""
        return list({w for w in words if self.known(w)})

    def edits1(self, word):
        """Return all edits that are one edits away from word."""
        letters = "abcdefghijklmnopqrstuvwxyz "
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        # deletes = [L + R[1:] for L, R in splits if R]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        # inserts = [L + c + R for L, R in splits for c in letters]
        # return set(deletes + replaces_vowels + inserts)
        return set(replaces)

    def edits2(self, word):
        """Return all edits that are two edits away from word."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def edits3(self, word):
        """Return all edits that are three edits away from word."""
        return (
            e3
            for e1 in self.edits1(word)
            for e2 in self.edits1(e1)
            for e3 in self.edits1(e2)
        )

    def choose_best_candidate(self, candidates):
        """Return the best candidate for correction."""
        return candidates[0]
