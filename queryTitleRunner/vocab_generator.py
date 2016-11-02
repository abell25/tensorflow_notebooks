from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import re
import os
import itertools
import sys
import time
import math

from tensorflow.models.rnn.translate import seq2seq_model

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

class VocabGenerator:
    def __init__(self, tokenizer=None, special_chars=['_PAD_ID', '_GO_ID', '_EOS_ID', '_UNK_ID']):
        if tokenizer is None:
            tokenizer = lambda s: [w for w in re.split(" +", s.lower()) if len(w.strip()) > 0]
        self.tokenizer = tokenizer 
        self.special_chars = special_chars
        
        self.sent_set = set()
        self.word_counts = {}
            
    def processSent(self, sent):
        if hash(sent) in self.sent_set:
            return
        
        self.sent_set.add(hash(sent))
        for word in self.tokenizer(sent):
            if len(word.strip()) > 0:
                self.word_counts[word] = 0 if word not in self.word_counts else self.word_counts[word]+1
            
    def generateVocab(self, sents=[], vocab_size=10000):
        for sent in sents:
            self.processSent(sent)

        vocab_remaining = vocab_size - len(self.special_chars)
        top_word_counts = sorted([(w, self.word_counts[w]) for w in self.word_counts], key=lambda x: x[1], reverse=True)[:vocab_remaining]
        self.vocab = self.special_chars + [w for w,c in top_word_counts]
        self.vocab_lookup = {w: i for i,w in enumerate(self.vocab)}
        return self.vocab
    
    def setVocab(self, vocab):
        self.vocab = vocab
        self.vocab_lookup = {w: i for i,w in enumerate(self.vocab)}
        
    def saveVocab(self, vocab_file, vocab=None):
        vocab = vocab if vocab is not None else self.vocab
        with open(vocab_file, "w") as out:
            for w in vocab:
                print(w, file=out)
                
    def loadVocab(self, vocab_file):
        vocab = [line.strip("\n") for line in open(vocab_file)]
        self.setVocab(vocab)
        return vocab
    
    def sentToIndexes(self, sent):
        #return [self.vocab_lookup[w] if w in self.vocab_lookup else -1 for w in self.tokenizer(sent)]
        words = self.tokenizer(sent)
        indexes = [self.vocab_lookup[w] if w in self.vocab_lookup else self.vocab_lookup['_UNK_ID'] for w in  words]
        return indexes if len(indexes) > 0 else [self.vocab_lookup['_UNK_ID']]
    
    def indexesToSents(self, indexes):
        return " ".join([self.vocab[i] for i in indexes])

