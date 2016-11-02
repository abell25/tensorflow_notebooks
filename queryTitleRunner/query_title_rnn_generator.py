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
import random

from tensorflow.models.rnn.translate import seq2seq_model
from vocab_generator import VocabGenerator
from query_title_loader import QueryTitleLoader

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
bucket_id=0

class QueryTitleRNNGenerator:
    def __init__(self, checkpoint_dir, train_loader, test_loader, num_layers, size, src_vocab_size, dest_vocab_size, max_grad_clip, batch_size, learning_rate, learning_rate_decay_factor):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        
        self.num_layers = num_layers
        self.size = size
        self.src_vocab_size = src_vocab_size
        self.dest_vocab_size = dest_vocab_size
        self.max_grad_clip = max_grad_clip
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        
        self.steps_per_checkpoint = 20
        
        self.session = tf.Session()
        
        
    def load_data(self, data_dir):
        self.queryTitlePairLoader = QueryTitleLoader(data_dir, max_query_len=6, max_title_len=16)
        queryTitlePairLoader.generateVocab(num_examples=10000, query_vocab_size=self.src_vocab_size, title_vocab_size=self.dest_vocab_size)
        return [self.queryTitlePairLoader.getQueryTitlesIntBatch(10000)]
       
    def create_model(self, session, forward_only):
        model = seq2seq_model.Seq2SeqModel(
            self.src_vocab_size,
            self.dest_vocab_size,
            [(6,16)],
            self.size,
            self.num_layers,
            self.max_grad_clip,
            self.batch_size,
            self.learning_rate,
            self.learning_rate_decay_factor,
            forward_only=forward_only,
            dtype=tf.float32)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return model
    
    def train(self):
        #with tf.Session() as sess:
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                #sess = self.session
                model = self.create_model(sess, False)
                self.model = model

                #train_set = self.train_loader.getQueryTitlesIntBatch(20000000)
                #test_set = self.test_loader.getQueryTitlesIntBatch(5000000)

                # This is the training loop.
                step_time, loss = 0.0, 0.0
                current_step = 0
                self.previous_losses = []
                self.train_perplexity = []
                self.test_perplexity = []
                self.learning_rates = []
                self.step_sizes = []
                while True:
                    start_time = time.time()
                    train_set = [self.train_loader.getQueryTitlesIntBatch(self.batch_size+10)]
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
                    _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
                    step_time += (time.time() - start_time) / self.steps_per_checkpoint
                    loss += step_loss / self.steps_per_checkpoint
                    current_step += 1

                    self.train_perplexity.append(math.exp(float(loss)) if loss < 300 else float("inf"))
                    if current_step % self.steps_per_checkpoint == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        print("global step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}".format(
                                model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                        #Print some example outputs
                        batch = self.test_loader.getQueryTitlesBatch(self.batch_size+10)
                        test_set = [batch[k] for k in [random.choice(range(len(batch))) for _ in range(5)]]
                        results = self.decodeWithModel(sess, model, [q for q,t in test_set])
                        print("results: ")
                        for t, (q,new_t) in zip([t for q,t in test_set], results):
                            print("[{:25}] [{:40}] [{:40}]".format(q[:25], t[:40], new_t[:40]))

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(self.previous_losses) > 2 and loss > max(self.previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)

                    self.learning_rates.append(model.learning_rate.eval())
                    self.step_sizes.append(model.global_step.eval())

                    self.previous_losses.append(loss)

                    checkpoint_path = os.path.join(self.checkpoint_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    step_time, loss = 0.0, 0.0

                    # Run evals on development set and print their perplexity.
                    test_set = [self.test_loader.getQueryTitlesIntBatch(self.batch_size+10)]
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    self.test_perplexity.append(eval_ppx)
                    if current_step % 5 == 0:print("eval: perplexity {:.2f}".format(eval_ppx))
                
        
    def decodeQuery(self, indexes):
        return self.train_loader.queryVocabGenerator.indexesToSents(indexes)
    
    def decodeTitle(self, indexes):
        return self.train_loader.titleVocabGenerator.indexesToSents(indexes)
        
    def encodeQuery(self, query):
        return self.train_loader.queryVocabGenerator.sentToIndexes(query)
        
    def encodeTitle(self, title):
        return self.train_loader.titleVocabGenerator.sentToIndexes(title)
       
    def decodeWithModel(self, sess, model, sentences):
        batch_size = model.batch_size
        model.batch_size = 1  

        output_sents = []
        for sentence in sentences:
            #print("sentence: {}".format(sentence))
            token_ids = self.encodeQuery(sentence)

            #print("sent: {}, token_ids: {}, bucket_id: {}".format(sentence, token_ids, bucket_id))
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            
            #print("token_ids:{}, encoder_inputs:{}, decoder_inputs:{}, target_weights:{}, output_logits:{}".format(
            #        token_ids, len(encoder_inputs), len(decoder_inputs), len(target_weights), len(output_logits)))
            #print("item sizes: token_ids:{}, encoder_inputs:{}, decoder_inputs:{}, target_weights:{}, output_logits:{}".format(
            #        token_ids[0], encoder_inputs, decoder_inputs, target_weights, output_logits[0].shape))

            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]

            output_sents.append((sentence, self.decodeTitle(outputs)))
        model.batch_size = batch_size
        return output_sents
        
    def decode(self, sentences):
        sess = self.session 
        if not hasattr(self, 'model'):
            print("reloading model....")
            self.model = self.create_model(sess, True)
        return self.decodeWithModel(sess, self.model, sentences)

    def self_test(self):
        """Test the translation model."""
        with tf.Session() as sess:
            print("Self-test for neural translation model.")
            # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
            model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
            sess.run(tf.initialize_all_variables())

            # Fake data set for both the (3, 3) and (6, 6) bucket.
            data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])], [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
            for _ in xrange(5):  # Train the fake model for 5 steps.
                bucket_id = random.choice([0, 1])
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
                model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)


if __name__ == "__main__":
    
    # In[5]:
    
    checkpoint_dir="/home/ubuntu/data/queryClickPairs/checkpoints"
    data_train_dir="/home/ubuntu/data/queryClickPairs/train"
    data_test_dir="/home/ubuntu/data/queryClickPairs/test"
    vocab_dir="/home/ubuntu/data/queryClickPairs/vocab"
    
    
    rnn_size = 1024
    num_layers = 3
    batch_size = 128
    src_vocab_size = 10000
    dest_vocab_size = 10000
    max_grad_clip = 5.0
    learning_rate = 0.50
    learning_rate_decay_factor = 0.99
    
    buckets = [(6, 16)] #[(5, 10), (10, 15), (20, 25), (40, 50)]
    
    
    # In[6]:
    
    queryTitlePairLoader_train = QueryTitleLoader(data_train_dir, vocab_dir, max_query_len=5, max_title_len=15)
    queryTitlePairLoader_test = QueryTitleLoader(data_test_dir, vocab_dir, max_query_len=5, max_title_len=15)
    
    #queryVocab, titleVocab = queryTitlePairLoader_train.generateVocab(num_examples=10000000, query_vocab_size=src_vocab_size, title_vocab_size=dest_vocab_size)
    #queryTitlePairLoader_test.setVocab(queryVocab, titleVocab)
    #queryTitlePairLoader_train.saveVocab()
    
    queryTitlePairLoader_train.loadVocab()
    queryTitlePairLoader_test.loadVocab()
    
    
    # In[ ]:
    
    queryTitleRNNGenerator = QueryTitleRNNGenerator(checkpoint_dir, queryTitlePairLoader_train, queryTitlePairLoader_test, num_layers, rnn_size, src_vocab_size, dest_vocab_size, 
                                                    max_grad_clip, batch_size, learning_rate, learning_rate_decay_factor)
    
    
    # In[ ]:
    
    queryTitleRNNGenerator.train()
    
     
