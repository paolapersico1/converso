"""Module to perform the speech correction."""
from dataclasses import dataclass
import logging
import math
from os import path
import re

from nltk import WhitespaceTokenizer
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams
import pandas as pd

from .const import NGRAMS_DIR
from .intent_recognition.const import NUMBER_DICT
from .intent_recognition.data_acquisition import load_synthetic_dataset
from .intent_recognition.data_preprocessing import preprocess_text

_LOGGER = logging.getLogger(__name__)


@dataclass
class Edit:
    """Edit class."""

    previous: str
    new: str
    edits: int
    bigram_ll: float


class SpeechCorrector:
    """Speech corrector for text commands."""

    def __init__(self) -> None:
        """Initialize the corrector."""
        unigram_df = pd.read_csv(
            path.join(NGRAMS_DIR, "it.word.top_unigrams_prob.csv"),
            index_col=["ngram"],
            usecols=["ngram", "frequency", "prob"],
        )
        bigram_df = pd.read_csv(
            path.join(NGRAMS_DIR, "it.word.top_bigrams_prob.csv"),
            index_col=["word1", "word2"],
            usecols=["word1", "word2", "frequency", "prob"],
        )
        unigram_df = unigram_df.sort_index(level=unigram_df.index.names)
        unigram_df = unigram_df[~unigram_df.index.duplicated()]
        bigram_df = bigram_df.sort_index(level=bigram_df.index.names)
        bigram_df = bigram_df[~bigram_df.index.duplicated()]
        df = load_synthetic_dataset()
        domain_vocab: set[str] = set()
        df["Text"].apply(preprocess_text).apply(domain_vocab.update)
        for num in NUMBER_DICT:
            domain_vocab.add(num)
        domain_vocab.remove("custom_name")
        self.improv_threshold = [10, 20, 30]
        self.lambda1 = 1 - 1e-8
        self.plausibility_threshold = 1e-8
        self.domain_vocab = pd.Index(domain_vocab).sort_values()
        self.vocab = unigram_df.index.sort_values()
        self.custom_ngrams: set = set()
        self.custom_words: set = set()
        self.unigram_df = unigram_df
        self.bigram_df = bigram_df
        self.unigram_count = unigram_df["frequency"].sum()
        self.bigram_count = bigram_df["frequency"].sum()
        self.cached_cond_prob: dict = {}
        self.tk = WhitespaceTokenizer()
        self.domain_bigrams: set = set()
        for sentence in df["Text"]:
            sequence = list(pad_both_ends(preprocess_text(sentence), n=2))
            self.domain_bigrams = self.domain_bigrams.union(set(ngrams(sequence, 2)))

    def add_custom(self, custom_set):
        """Add custom ngrams."""
        custom_set = {re.search("text='(.+?)'", c).group(1).lower() for c in custom_set}
        new_words = set(
            sum([preprocess_text(custom_name) for custom_name in custom_set], [])
        )
        self.custom_words = new_words
        self.domain_vocab = self.domain_vocab.union(new_words)
        self.vocab = self.vocab.union(new_words)
        for c in custom_set:
            sequence = preprocess_text(c)
            if len(sequence) > 1:
                self.custom_ngrams = self.custom_ngrams.union(set(ngrams(sequence, 2)))
            for bigram in self.domain_bigrams.copy():
                if "custom_name" in bigram[0]:
                    self.domain_bigrams.add((sequence[-1], bigram[1]))
                elif "custom_name" in bigram[1]:
                    self.domain_bigrams.add((bigram[0], sequence[0]))

    def legit_edits(self, edits, vocab):
        """Return edits in the domain."""
        result = []
        for edit in edits:
            legit = True
            edit_tokens = self.tk.tokenize(edit)
            for edit_token in edit_tokens:
                if edit_token not in vocab and edit_token not in ("<s>", "</s>"):
                    legit = False
                    break
            if legit:
                result.append(edit)

        return result

    def raw_count(self, ngram):
        """Compute raw count."""
        if isinstance(ngram, str):
            if ngram in self.custom_words or ngram in self.domain_vocab:
                return self.unigram_df.frequency.max()
            df = self.unigram_df
        else:
            if ngram in self.domain_bigrams.union(self.custom_ngrams):
                return self.bigram_df.frequency.max()
            df = self.bigram_df

        if ngram in df.index:
            return df.loc[ngram, "frequency"]

        return df.frequency.min()

    def prob(self, ngram):
        """Compute joint probability of ngram."""
        if isinstance(ngram, str):
            if ngram in self.custom_words or ngram in self.domain_vocab:
                return self.unigram_df.prob.max()
            df = self.unigram_df
        else:
            if ngram in self.domain_bigrams.union(self.custom_ngrams):
                return self.bigram_df.prob.max()
            df = self.bigram_df

        if ngram in df.index:
            result = df.loc[ngram, "prob"]
        else:
            result = df.prob.min()
        return result

    def conditional_prob(self, left, right):
        """Compute conditional probability with raw count."""
        if (left, right) in self.cached_cond_prob:
            # print("Cache hit")
            result = self.cached_cond_prob[(left, right)]
        else:
            num = self.raw_count((right, left))
            den = self.raw_count(right)
            cond_prob = float(num) / den

            result = (1 - self.lambda1) * cond_prob + self.lambda1 * self.prob(left)
            self.cached_cond_prob[(left, right)] = result

        return result

    def sentence_LL(self, sequence):
        """Compute the bigram log likelihood of a sequence."""
        ll = 0
        n_grams = list(ngrams(sequence, 2))
        for pre_word, post_word in n_grams:
            cond_prob = self.conditional_prob(post_word, pre_word)
            ll = ll + math.log(cond_prob)

        # _LOGGER.debug(str(sequence) + " LL: " + str(ll))
        return ll

    def generate_ngram_candidates(self, ngram, sentence, vocab, edits_num, edits):
        """Generate possible error corrections for ngrams."""
        candidates: list[Edit] = []

        for edit in self.legit_edits(edits, vocab):
            possible_sequence = self.tk.tokenize(sentence.replace(ngram, edit))
            possible_sequence = list(pad_both_ends(possible_sequence, n=2))
            bigram_ll = self.sentence_LL(possible_sequence)
            candidates.append(
                Edit(previous=ngram, new=edit, edits=edits_num, bigram_ll=bigram_ll)
            )
        return candidates

    def detect_errors(self, sequence):
        """Detect possible errors in a sequence."""
        errors = []
        for bigram in list(ngrams(sequence, 2)):
            prob = self.prob(bigram)
            # _LOGGER.debug(str(bigram) + " probability: " + str(prob))
            if prob < self.plausibility_threshold:
                bigram_as_str = " ".join(bigram)
                errors.append(bigram_as_str)

        errors = errors + [
            unigram for unigram in sequence if unigram not in self.domain_vocab
        ]
        _LOGGER.debug("Detected errors: " + str(errors))
        return errors

    def correct(self, sentence):
        """Return a corrected sentence."""

        sentence = sentence.replace(",", "")
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("!", "")
        sequence = preprocess_text(sentence)
        sentence = " ".join(sequence)

        errors = self.detect_errors(sequence)
        sequence = list(pad_both_ends(sequence, n=2))

        while errors:
            error = errors[0]
            errors.remove(error)

            original_ll = self.sentence_LL(list(pad_both_ends(sequence, n=2)))
            # _LOGGER.debug("Original bigram LL: " + str(original_ll))

            if error in sentence:
                _LOGGER.debug("FIXING ERROR: " + error)
                # _LOGGER.debug("Generating first candidates")
                first_edits = self.edits1(error)
                candidates = self.generate_ngram_candidates(
                    error, sentence, self.domain_vocab, edits_num=1, edits=first_edits
                )
                sentence, changed = self.replace_with_best_candidate(
                    sentence, candidates, 0, original_ll
                )
                if not changed:
                    # _LOGGER.debug("Generating second candidates")
                    second_edits = self.edits2(first_edits)
                    candidates = self.generate_ngram_candidates(
                        error, sentence, self.vocab, edits_num=1, edits=first_edits
                    )
                    candidates = candidates + self.generate_ngram_candidates(
                        error,
                        sentence,
                        self.domain_vocab,
                        edits_num=2,
                        edits=second_edits,
                    )
                    sentence, changed = self.replace_with_best_candidate(
                        sentence, candidates, 1, original_ll
                    )
                    if not changed:
                        # _LOGGER.debug("Generating third candidates")
                        candidates = self.generate_ngram_candidates(
                            error, sentence, self.vocab, edits_num=2, edits=second_edits
                        )
                        sentence, changed = self.replace_with_best_candidate(
                            sentence, candidates, 2, original_ll
                        )
                if changed:
                    # _LOGGER.debug("New sentence: " + sentence)
                    sequence = self.tk.tokenize(sentence)

        sentence = sentence.replace("<s> ", "")
        sentence = sentence.replace(" </s>", "")
        _LOGGER.debug("Corrected sentence: " + sentence)

        return sentence

    def edits1(self, word, with_space=True):
        """Return all edits that are one edit away from word."""
        letters = "abcdefghijklmnopqrstuvwxyz'"
        if with_space:
            letters = letters + " "
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R and (not R[0].isnumeric())]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [
            L + c + R[1:]
            for L, R in splits
            if R and (not R[0].isnumeric())
            for c in letters
        ]
        inserts = [L + c + R for L, R in splits for c in letters]

        result = set(deletes + replaces + inserts + transposes)
        return result

    def edits2(self, words):
        """Return all edits that are two edit away from words."""
        return {e2 for e1 in words for e2 in self.edits1(e1, False)}

    def replace_with_best_candidate(self, sentence, candidates, stage, original_ll):
        """Return the best candidate for correction."""
        changed = False
        if candidates:
            best_candidate = candidates[0]
            best_ll_bi = candidates[0].bigram_ll
            for candidate in candidates[1:]:
                if candidate.bigram_ll > best_ll_bi:
                    best_candidate = candidate
                    best_ll_bi = candidate.bigram_ll

            # _LOGGER.debug("Best candidate: " + str(best_candidate))
            if (best_ll_bi - original_ll) > self.improv_threshold[stage] / len(
                sentence
            ):
                sentence = sentence.replace(best_candidate.previous, best_candidate.new)
                changed = True
        return sentence, changed
