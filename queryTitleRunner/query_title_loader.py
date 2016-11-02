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
from vocab_generator import VocabGenerator

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class QueryTitleLoader:
    def __init__(self, path, vocab_dir, max_query_len=6, max_title_len=16):
        self.path = path
        self.vocab_dir = vocab_dir
        self.max_query_len = max_query_len
        self.max_title_len = max_title_len
        self.getQueryTitles()
        
        self.queryVocabGenerator, self.titleVocabGenerator = VocabGenerator(), VocabGenerator()

    @staticmethod
    def normalize(line):
        line = line.lower().strip()
        line = re.sub("[-/\\\\]", " ", line)
        line = re.sub("[^-\w\d \.\t]", " ", line)
        line = re.sub("  +", " ", line)
        return line

    @staticmethod
    def parseLine(line):
        vals = line.strip("\n ").split("\t")
        query, title = vals[0], vals[2]
        return (query, title)

    def queryTitleFilter(self, query, title):
        negative_match = "-\(|^-\w| -\w"
        chars_to_reject = "[\(\)\[\]\"]"
        num_only = "^-?[0-9\.]+$"
        any_number = "-?[0-9\.]{2,}"
        if len(query.split(" ")) > self.max_query_len or len(title.split(" ")) > self.max_title_len:
            return False
        if len(query.split(" ")) < 1 or len(query.strip()) == 0:
            return False
        if len(re.findall("|".join([negative_match, chars_to_reject, num_only, any_number]), query)) > 0:
            return False   
        return True
    
    @staticmethod
    def writeLines(ar, filename):
        with open(filename, "w") as out:
             for i, x in enumerate(ar):
                print(x, file=out)
                
    def getQueryTitles(self):
        queryTitlePairsRaw = (QueryTitleLoader.parseLine(line) for f in os.listdir(self.path) for line in open(os.path.join(self.path, f), "r"))
        queryTitlePairs = ((QueryTitleLoader.normalize(q), QueryTitleLoader.normalize(t)) for q,t in queryTitlePairsRaw if self.queryTitleFilter(q,t))
        self.queryTitlePairs = queryTitlePairs
        return self.queryTitlePairs
    
    def getQueryTitlesInts(self):
        return (self.queryTitleToIndexes(q, t) for q,t in self.getQueryTitles())
    
    def getQueryTitlesBatch(self, num_records): 
        batch = []
        for k in range(num_records):
            try:
                batch.append(next(self.queryTitlePairs))
            except StopIteration:
                self.getQueryTitles()
        return batch
    
    def queryTitleToIndexes(self, query, title):
        return (self.queryVocabGenerator.sentToIndexes(query), self.titleVocabGenerator.sentToIndexes(title))
    
    def getQueryTitlesIntBatch(self, num_records): 
        batch = self.getQueryTitlesBatch(num_records)
        return [self.queryTitleToIndexes(q,t) for q,t in batch]
    
    def generateVocab(self, num_examples=1000000, query_vocab_size=10000, title_vocab_size=10000):        
        for q,t in self.getQueryTitlesBatch(num_examples):
            self.queryVocabGenerator.processSent(q)
            self.titleVocabGenerator.processSent(t)

        self.q_vocab = self.queryVocabGenerator.generateVocab(vocab_size=query_vocab_size)
        self.t_vocab = self.titleVocabGenerator.generateVocab(vocab_size=title_vocab_size)
        
        return (self.q_vocab, self.t_vocab)
    
    def setVocab(self, queryVocab, titleVocab):
        self.q_vocab = self.queryVocabGenerator.setVocab(queryVocab)
        self.t_vocab = self.titleVocabGenerator.generateVocab(titleVocab)
        
    def saveVocab(self):
        self.queryVocabGenerator.saveVocab(os.path.join(self.vocab_dir, "query_vocab.txt"))
        self.titleVocabGenerator.saveVocab(os.path.join(self.vocab_dir, "title_vocab.txt"))
    
    def loadVocab(self):
        self.q_vocab = self.queryVocabGenerator.loadVocab(os.path.join(self.vocab_dir, "query_vocab.txt"))
        self.t_vocab = self.titleVocabGenerator.loadVocab(os.path.join(self.vocab_dir, "query_vocab.txt"))
        

