#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
from math import log
import sys
import copy
import string

from hw2.lm import xrange

# if word not in self.ngram_counts:
#     if 'UNK' in self.ngram_counts[prev]:
#         return log(self.ngram_counts[prev]['UNK'], 2)
#     else:
#         return log(1 / len(self.ngram_counts.keys()))

def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams
    ngrams of tuple form: ((previous wordS!), target word)
    """
    for i in range(n):
        tokens.insert(0, "<START>")

    print(tokens)
    #tokens = tokens.insert(n-1)*['<START>']+tokens+["END_OF_SENTENCE"]
    #ngrams = zip(*[tokens[i:] for i in range(n)])
    return #[" ".join(ngram) for ngram in ngrams]

if __name__ == "__main__":
    import string

    # sentence_list = ["'my', 'name', 'is', 'Jerry'", "my name is Josh.", "my name is Bob", "my name is Steve."]
    # ngram_counter = dict()
    #
    # print(get_ngrams(2, sentence_list[0]))

    previous = ["hi", "my ", "name", "is"]
    ngram_len = 3

    print(' '.join(previous[-(ngram_len-1):]))

    for i in xrange(len(previous)):
       print(previous[i], previous[:i])
    # dnames = ["brown"]
    # print("-----------------------")
    # print(dname)
    # data = read_texts("hw2/data/corpora.tar.gz", dname)



    # for sentence in sentence_list:
    #     ngrams = get_ngrams(2, sentence)
    #
    #     for ngram in ngrams:
    #         if ngram in ngram_counter:
    #             ngram_counter[ngram] += 1.0
    #         else:
    #             ngram_counter[ngram] = 1.0
    # print("values")
    # print(ngram_counter.values())
    # print("keys")
    # print(ngram_counter.keys())
    #omit words in dictionary with values less than 2
    # for key in ngram_counter:
    #     print("PRINTING")
    #     if(ngram_counter[key] < 2) {
    #
    #     } else {
    #     pass
    #     }





