#!/usr/bin/env python3
# Student name: Richard Yan
# Student number: 1005731193
# UTORid: yanrich2

from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    return wn.synsets(sentence[word_index].lemma)[0]


def lesk(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    senses = wn.synsets(sentence[word_index].lemma)
    best_sense = senses[0]
    if len(senses) == 1:
        return best_sense
    best_score = 0
    context_lst = [x.wordform for x in sentence]
    context = stop_tokenize(' '.join(context_lst))
    context_dict = Counter(context)
    for sense in senses:
        examples_str = ' '.join(sense.examples())
        signature_str = sense.definition() + ' ' + examples_str
        signature = stop_tokenize(signature_str)
        signature_dict = Counter(signature)
        score = 0
        for w in context_dict:
            if w in signature_dict:
                score += min(context_dict[w], signature_dict[w])
        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense


def lesk_ext(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    senses = wn.synsets(sentence[word_index].lemma)
    best_sense = senses[0]
    if len(senses) == 1:
        return best_sense
    best_score = 0
    context_lst = [x.wordform for x in sentence]
    context = stop_tokenize(' '.join(context_lst))
    context_dict = Counter(context)
    for sense in senses:
        signature_str = sense.definition()
        signature_str += ' ' + ' '.join(sense.examples())
        for hyponym in sense.hyponyms():
            signature_str += ' ' + hyponym.definition()
            signature_str += ' ' + ' '.join(hyponym.examples())
        for holonym in sense.member_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.part_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.substance_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for meronym in sense.member_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.substance_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.part_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        signature = stop_tokenize(signature_str)
        signature_dict = Counter(signature)
        score = 0
        for w in context_dict:
            if w in signature_dict:
                score += min(context_dict[w], signature_dict[w])
        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense


def lesk_cos(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    senses = wn.synsets(sentence[word_index].lemma)
    best_sense = senses[0]
    if len(senses) == 1:
        return best_sense
    best_score = 0
    context_lst = [x.wordform for x in sentence]
    context = stop_tokenize(' '.join(context_lst))
    context_dict = Counter(context)
    for sense in senses:
        signature_str = sense.definition()
        signature_str += ' ' + ' '.join(sense.examples())
        for hyponym in sense.hyponyms():
            signature_str += ' ' + hyponym.definition()
            signature_str += ' ' + ' '.join(hyponym.examples())
        for holonym in sense.member_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.part_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.substance_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for meronym in sense.member_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.substance_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.part_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        signature = stop_tokenize(signature_str)
        signature_dict = Counter(signature)
        vec_vocab = context_dict.keys() | signature_dict.keys()
        context_vec = [context_dict[x] for x in vec_vocab]
        signature_vec = [signature_dict[x] for x in vec_vocab]
        score = np.dot(context_vec, signature_vec) / (norm(context_vec) * norm(signature_vec))
        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense


def lesk_cos_onesided(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    senses = wn.synsets(sentence[word_index].lemma)
    best_sense = senses[0]
    if len(senses) == 1:
        return best_sense
    best_score = 0
    context_lst = [x.wordform for x in sentence]
    context = stop_tokenize(' '.join(context_lst))
    context_dict = Counter(context)
    for sense in senses:
        signature_str = sense.definition()
        signature_str += ' ' + ' '.join(sense.examples())
        for hyponym in sense.hyponyms():
            signature_str += ' ' + hyponym.definition()
            signature_str += ' ' + ' '.join(hyponym.examples())
        for holonym in sense.member_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.part_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.substance_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for meronym in sense.member_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.substance_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.part_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        signature = stop_tokenize(signature_str)
        signature_dict = Counter(signature)
        vec_vocab = context_dict.keys()
        context_vec = [*context_dict.values()]
        signature_vec = [signature_dict[x] for x in vec_vocab]
        denominator = norm(context_vec) * norm(signature_vec) + 1e-7
        score = np.dot(context_vec, signature_vec) / denominator
        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense


def lesk_w2v(sentence: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    senses = wn.synsets(sentence[word_index].lemma)
    best_sense = senses[0]
    if len(senses) == 1:
        return best_sense
    best_score = 0
    context_lst = [x.wordform for x in sentence]
    context = stop_tokenize(' '.join(context_lst))
    for sense in senses:
        signature_str = sense.definition()
        signature_str += ' ' + ' '.join(sense.examples())
        for hyponym in sense.hyponyms():
            signature_str += ' ' + hyponym.definition()
            signature_str += ' ' + ' '.join(hyponym.examples())
        for holonym in sense.member_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.part_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for holonym in sense.substance_holonyms():
            signature_str += ' ' + holonym.definition()
            signature_str += ' ' + ' '.join(holonym.examples())
        for meronym in sense.member_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.substance_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        for meronym in sense.part_meronyms():
            signature_str += ' ' + meronym.definition()
            signature_str += ' ' + ' '.join(meronym.examples())
        signature = stop_tokenize(signature_str)
        context_vec_lst = []
        for word in context:
            if ' ' in word:
                temp = word.replace(' ', '-')
                if temp in vocab:
                    vec = word2vec[vocab[temp]]
                elif temp.lower() in vocab:
                    vec = word2vec[vocab[temp.lower()]]
                else:
                    words = word.split()
                    temp_vec = []
                    for w in words:
                        if w in vocab:
                            temp_vec.append(word2vec[vocab[w]])
                        elif w.lower() in vocab:
                            temp_vec.append(word2vec[vocab[w.lower()]])
                        else:
                            temp_vec.append(np.zeros_like(word2vec[0]))
                    vec = np.mean(temp_vec, axis=0)
            else:
                if word in vocab:
                    vec = word2vec[vocab[word]]
                elif word.lower() in vocab:
                    vec = word2vec[vocab[word.lower()]]
                else:
                    vec = np.zeros_like(word2vec[0])
            context_vec_lst.append(vec)
        signature_vec_lst = []
        for word in signature:
            if ' ' in word:
                temp = word.replace(' ', '-')
                if temp in vocab:
                    vec = word2vec[vocab[temp]]
                elif temp.lower() in vocab:
                    vec = word2vec[vocab[temp.lower()]]
                else:
                    words = word.split()
                    temp_vec = []
                    for w in words:
                        if w in vocab:
                            temp_vec.append(word2vec[vocab[w]])
                        elif w.lower() in vocab:
                            temp_vec.append(word2vec[vocab[w.lower()]])
                        else:
                            temp_vec.append(np.zeros_like(word2vec[0]))
                    vec = np.mean(temp_vec, axis=0)
            else:
                if word in vocab:
                    vec = word2vec[vocab[word]]
                elif word.lower() in vocab:
                    vec = word2vec[vocab[word.lower()]]
                else:
                    vec = np.zeros_like(word2vec[0])
            signature_vec_lst.append(vec)
        context_vec = np.mean(context_vec_lst, axis=0)
        signature_vec = np.mean(signature_vec_lst, axis=0)
        score = np.dot(context_vec, signature_vec) / (norm(context_vec) * norm(signature_vec))
        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
