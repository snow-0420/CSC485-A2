#!/usr/bin/env python3
# Student name: Richard Yan
# Student number: 1005731193
# UTORid: yanrich2

import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    max_depth = -1
    max_synset = None
    for synset in wn.all_synsets():
        if synset.max_depth() > max_depth:
            max_depth = synset.max_depth()
            max_synset = synset
    for path in max_synset.hypernym_paths():
        print("Path: {0}, depth: {1}".format(path, len(path)))


def superdefn(s: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    word = wn.synset(s)
    sdefn = [x for x in word_tokenize(word.definition())]
    for x in word.hypernyms():
        sdefn += word_tokenize(x.definition())
    for x in word.hyponyms():
        sdefn += word_tokenize(x.definition())
    return sdefn


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    stopword = stopwords.words('english')
    return [word for word in word_tokenize(s) if word.lower() not in stopword and word not in punctuation]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
