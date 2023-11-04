#!/usr/bin/env python3
# Student name: Richard Yan
# Student number: 1005731193
# UTORid: yanrich2

import typing as T
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import torch
from torch import Tensor
from torch.linalg import norm

from tqdm import tqdm, trange

from q1 import mfs
from wsd import (batch_evaluate, load_bert, run_bert, load_eval, load_train,
                 WSDToken)


def gather_sense_vectors(corpus: T.List[T.List[WSDToken]],
                         batch_size: int = 32) -> T.Dict[str, Tensor]:
    """Gather sense vectors using BERT run over a corpus.

    As with A1, it is much more efficient to batch the sentences up than it is
    to do one sentence at a time, and you can further improve (~twice as fast)
    if you sort the corpus by sentence length first. We've therefore started
    this function out that way for you, but you may implement the code in this
    function however you like.

    The procedure for this function is as follows:
    * Use run_bert to run BERT on each batch
    * Go through all of the WSDTokens in the input batch. For each one, if the
      token has any synsets assigned to it (check WSDToken.synsets), then add
      the BERT output vector to a list of vectors for that sense (**not** for
      the token!).
    * Once this is done for all batches, then for each synset that was seen
      in the corpus, compute the mean of all vectors stored in its list.
    * That yields a single vector associated to each synset; return this as
      a dictionary.

    The run_bert function will handle tokenizing the batch for BERT, including
    padding the tokenized sentences so that each one has the same length, as
    well as converting it to a PyTorch tensor that lives on the GPU. It then
    runs BERT on it and returns the output vectors from the top layer.

    An important point: the tokenizer will produce more tokens than in the
    original input, because sometimes it will split one word into multiple
    pieces. BERT will then produce one vector per token. In order to
    produce a single vector for each *original* word token, so that you can
    then use that vector for its various synsets, you will need to align the
    output tokens back to the originals. You will then sometimes have multiple
    vectors for a single token in the input data; take the mean of these to
    yield a single vector per token. This vector can then be used like any
    other in the procedure described above.

    To provide the needed information to compute the token-word alignments,
    run_bert returns an offset mapping. For each token, the offset mapping
    provides substring indices, indicating the position of the token in the
    original word (or [0, 0] if the token doesn't correspond to any word in the
    original input, such as the [CLS], [SEP], and [PAD] tokens). You can
    inspect the returned values from run-bert in a debugger and/or try running
    the tokenizer on your own test inputs. Below are a couple examples, but
    keep in mind that these are provided purely for illustrative purposes
    and your actual code isn't to call the tokenizer directly itself.
        >>> from wsd import load_bert
        >>> load_bert()
        >>> from wsd import TOKENIZER as tknz
        >>> tknz('This is definitely a sentence.')
        {'input_ids': [101, 1188, 1110, 5397, 170, 5650, 119, 102],
         'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
        >>> out = tknz([['Multiple', ',', 'pre-tokenized', 'sentences', '!'], \
                        ['Much', 'wow', '!']], is_split_into_words=True, \
                        padding=True, return_offsets_mapping=True)
        >>> out.tokens(0)
        ['[CLS]', 'Multiple', ',', 'pre', '-', 'token', '##ized', 'sentences',
         '!', '[SEP]']
        >>> out.tokens(1)
        ['[CLS]', 'Much', 'w', '##ow', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]',
        '[PAD]']
        >>> out['offset_mapping']
        [[[0, 0], [0, 8], [0, 1], [0, 3], [3, 4], [4, 9], [9, 13], [0, 9],
         [0, 1], [0, 0]],
         [[0, 0], [0, 4], [0, 1], [1, 3], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]]]

    Args:
        corpus (list of list of WSDToken): The corpus to use.
        batch_size (int): The batch size to use.

    Returns:
        dict mapping synsets IDs to Tensor: A dictionary that can be used to
        retrieve the (PyTorch) vector for a given sense.
    """
    corpus = sorted(corpus, key=len)
    output = defaultdict(Tensor)
    for batch_n in trange(0, len(corpus), batch_size, desc='gathering',
                          leave=False):
        batch = corpus[batch_n:batch_n + batch_size]
        # extract str out from input word tokens
        batch_str = [[s.wordform for s in x] for x in batch]
        # run bert on those sentences
        vec_bert, offset_mapping = run_bert(batch_str)
        # for each sentence
        for id_sentence, sentence in enumerate(batch):
            # do token_word_aligning
            # each element in token_word_align represents the slice of that
            # word (original) in the tokenized sentence
            # so first element represent the first word (original)
            token_word_align = []
            id_last_word = 0
            for off in offset_mapping[id_sentence]:
                if off[1] != 0: #not in the form [0, 0] in offset_mapping
                    if off[0] == 0: #first token of that word (in [0, i] form)
                        token_word_align.append([id_last_word, id_last_word + 1])
                    else: #not first token of that word (in [i, j] form)
                        token_word_align[-1][1] += 1
                id_last_word += 1
            # for each word (token)
            for id_word_token, word_token in enumerate(sentence):
                # if it has synsets assigned to it
                if len(word_token.synsets) > 0:
                    # for each sense (synset)
                    for sense in word_token.synsets:
                        # if sense already as a key of the dict, concatenate the
                        # vector for further calculation
                        if sense in output:
                            output[sense] = torch.cat((output[sense], torch.mean(vec_bert[id_sentence, token_word_align[id_word_token][0]:token_word_align[id_word_token][1]], 0).reshape(-1, 1)), dim=1)
                        # calculate the vector for that sense (use mean because
                        # might be multi-tokens)
                        else:
                            output[sense] = torch.mean(vec_bert[id_sentence, token_word_align[id_word_token][0]:token_word_align[id_word_token][1]], 0).reshape(-1, 1)
    # for every sense (synset) in our return dict
    for sense in list(output.keys()):
        # calculate the vector as the mean of its list for that sense (synset)
        output[sense] = torch.mean(output[sense], dim=1)
    return output


def bert_1nn(batch: T.List[T.List[WSDToken]],
             indices: T.Iterable[T.Iterable[int]],
             sense_vectors: T.Mapping[str, Tensor]) -> T.List[T.List[Synset]]:
    """Find the best sense for specified words in a batch of sentences using
    the most cosine-similar sense vector.

    See the docstring for gather_sense_vectors above for examples of how to use
    BERT. You will need to run BERT on the input batch and associate a single
    vector for each input token in the same way. Once you've done this, you can
    compare the vector for the target word with the sense vectors for its
    possible senses, and then return the sense with the highest cosine
    similarity.

    In case none of the senses have vectors, return the most frequent sense
    (e.g., by just calling mfs(), which has been imported from q1 for you).

    **IMPORTANT**: When computing the cosine similarities and finding the sense
    vector with the highest one for a given target word, do not use any loops.
    Implement this aspect via matrix-vector multiplication and other PyTorch
    ops.

    Args:
        batch (list of list of WSDToken): The batch of sentences containing
            words to be disambiguated.
        indices (list of list of int): The indices of the target words in the
            batch sentences.
        sense_vectors: A dictionary mapping synset IDs to PyTorch vectors, as
            generated by gather_sense_vectors(...).

    Returns:
        predictions: The predictions of the correct sense for the given words.
    """
    # extract str out from input word tokens
    batch_str = [[s.wordform for s in x] for x in batch]
    # run bert on those sentences
    vec_bert, offset_mapping = run_bert(batch_str)
    # pre-initialize a tensor of size (# sentence in batch, # words in the
    # longest sentence in batch, size of the vector for computation)
    # (i.e. a vector assigned for each word token in batch)
    vec_token = torch.zeros(len(batch), len(batch[-1]), vec_bert.size(dim=2))
    # for each sentence
    for id_sentence, sentence in enumerate(batch):
        # do the token_word_aligning
        token_word_align = []
        id_last_word = 0
        for off in offset_mapping[id_sentence]:
            if off[1] != 0: #not in the form [0, 0] in offset_mapping
                if off[0] == 0: #first token of that word (in [0, i] form)
                    token_word_align.append([id_last_word, id_last_word + 1])
                else: #not first token of that word (in [i, j] form)
                    token_word_align[-1][1] += 1
            id_last_word += 1
        # get the vector for each word token in sentence (and store it in the
        # initialized tensor)
        for id_word_token, word_token in enumerate(sentence):
            # only do so when the word has senses assigned to it
            if len(word_token.synsets) > 0:
                # using mean to capture if the word is separated into multiple
                # tokens (will not affect the vector if only one token)
                vec_token[id_sentence, id_word_token] = torch.mean(vec_bert[id_sentence, token_word_align[id_word_token][0]:token_word_align[id_word_token][1]], 0)
    # initialize the list for prediction
    pred = []
    # for each sentence and target words of that sentence
    for id_sentence, id_targets in enumerate(indices):
        # initialize the list for prediction on the target words of that sentence
        pred_sentence = []
        # for each target word
        for id_t in id_targets:
            # get every senses of that target word
            senses = [sense for sense in wn.synsets(batch[id_sentence][id_t].lemma)]
            # get the vector of a sense if that sense is in the given senses_vectors
            # otherwise, fill the same size vector with 0
            vec_senses = [sense_vectors.get(synset.name(), torch.zeros(vec_bert.size(dim=2))) for synset in senses]
            # compute cosine similarity
            cos_sims = torch.div(torch.matmul(vec_token[id_sentence, id_t], torch.stack(vec_senses).T), norm(torch.stack(vec_senses), dim=1) * norm(vec_token[id_sentence, id_t]))
            # there are cases of dividing with 0, so replace all nan to 0
            # (since cosine similarity return value in [0, 1] as vectors
            # used for computation here has all positive values)
            cos_sims = torch.nan_to_num(cos_sims)
            # get the largest cosine similarity sense
            id_max = torch.argmax(cos_sims)
            if cos_sims[id_max] == 0: # if no senses has non-zero vector, use mfs
                sense = mfs(batch[id_sentence], id_t)
            else:
                sense = senses[id_max]
            pred_sentence.append(sense)
        pred.append(pred_sentence)
    return pred


if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    with torch.no_grad():
        load_bert()
        train_data = load_train()
        eval_data = load_eval()

        sense_vecs = gather_sense_vectors(train_data)
        batch_evaluate(eval_data, bert_1nn, sense_vecs)
