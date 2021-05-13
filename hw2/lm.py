#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
from curses import has_key
from math import log
import sys
import copy
import string

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.
        The corpus consists of a list of sentences."""
        self.vocab_words = {}
        for s in corpus:
            for word in s:
                if word in self.vocab_words:
                    self.vocab_words[word] += 1.0
                else:
                    self.vocab_words[word] = 1.0
                #keep counts of each word

        min_count = 2
        vocabulary = copy.deepcopy(self.vocab_words)
        for word in self.vocab_words:
            #if count is less than min_count, delete it from vocabulary
            if self.vocab_words[word] < min_count:
                del vocabulary[word]

        self.vocab_words = copy.deepcopy(vocabulary)

        # x = 0
        for s in corpus:
            self.fit_sentence(s)
            # x += 1
            # if x > 2:
            #     self.norm()
            #     exit()
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        numOOV = self.get_num_oov(corpus)
        return pow(2.0, self.entropy(corpus, numOOV))

    def get_num_oov(self, corpus):
        vocab_set = set(self.vocab())
        words_set = set(itertools.chain(*corpus))
        numOOV = len(words_set - vocab_set)
        return numOOV

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model supports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

class Ngram(LangModel):
    def __init__(self, ngram_size, unk_prob=0.0001):

        self.lunk_prob = log(unk_prob, 2)
        self.ngram_size = ngram_size
        self.ngram_counts = {}
        self.ngram_context_counts = {}

    def get_ngrams(self, num, tokens):
        for i in range(num-1):
            tokens.insert(0, "<START>")
        tokens.append("END_OF_SENTENCE")
        output = []
        for i in range(len(tokens) - num + 1):
            output.append(tokens[i:i + num])
        return output

    def unkify(self, sentence):
        for w in sentence:
            if w not in self.vocab_words:
                sentence = ['UNK' if i==w else i for i in sentence]
        return sentence

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        """
        Updates Language Model
        :param sentence: input text
        """
        sentence = self.unkify(sentence)
        ngrams = self.get_ngrams(self.ngram_size, sentence)
        for ngram in ngrams:
            prev, current = ' '.join(ngram[:-1]), ngram[-1]
            if prev not in self.ngram_counts:
                self.ngram_counts[prev] = {}
                self.ngram_context_counts[prev] = {}
            if current not in self.ngram_counts[prev]:
                self.ngram_counts[prev][current] = 1
            else:
                self.ngram_counts[prev][current] += 1


    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self):

        for prev in self.ngram_counts:
            total_count = 0.0
            for word in self.ngram_counts[prev]:
                total_count += self.ngram_counts[prev][word]
            for word in self.ngram_counts[prev]:
                self.ngram_counts[prev][word] = log((1 + self.ngram_counts[prev][word]) / (total_count + 1*len(self.vocab_words)),2) #saves ngram_counts[prev][word] as a percentage
            self.ngram_context_counts[prev] = log(1 / (total_count + 1 * len(self.vocab_words)), 2)

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV):

        word = self.unkify(word)
        previous = self.unkify(previous)
        word = ' '.join(word)
        previous_temp = previous
        for i in range(self.ngram_size-1):
            previous_temp.insert(0, "<START>")

        prev = ' '.join(previous_temp[-(self.ngram_size - 1):])

        if prev in self.ngram_counts:
            if word in self.ngram_counts[prev]: #both previous and word are in dictionary
                return self.ngram_counts[prev][word]
            else: #previous but not word
                return self.ngram_context_counts[prev]
        else:
            if word in self.vocab_words: #word but not previous
                return log((1-.0001)/len(self.vocab_words),2)
            else: #neither word nor previous
                return log(.0001,2)


    # required, the list of words the language model supports (including EOS)
    def vocab(self):
        return self.ngram_counts.keys()
