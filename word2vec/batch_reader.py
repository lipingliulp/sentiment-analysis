from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
import random

import sys

class ReviewReader:

    def __init__(self, reviews, window_size)
    
        self.reviews = reviews
        self.pos = 0
        self.window_size = window_size

    def read_batch():

        review = self.reviews[self.pos]
        self.pos = self.pos + 1

        data = review['text']
        atts = review['atts']

        batch_size = len(data) - self.contex_window * 2

        if config['use_sideinfo']:
            sidevec = np.tile(atts, (batch_size, 1))
        else:
            sidevec = np.ndarray(shape=(batch_size, 0), dtype=np.float32)

        batch = np.ndarray(shape=(batch_size, self.window_size * 2), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in xrange(batch_size):

            label_ind = i + self.window_size
            labels[i, 0] = data[label_ind]
        
            batch[i, 0 : self.window_size] = data[i : i + self.window_size]
            batch[i, self.window_size + 1 : ] = data[i + self.window_size + 1 : i + 2 * self.window_size + 1]

        return sidevec, batch, labels







class SkipGramReviewReader:

    def __init__(self, reviews, config)
        self.reviews = reviews
        self.pos = 0
        self.config = config

    def read_batch():

        num_skips = self.config['num_skips']
        skip_window = self.config['skip_window']
        review = self.reviews[self.pos]
        self.pos = self.pos + 1

        data = review['text']
        atts = review['atts']

        batch_size = len(data) * num_skips 
        data_index = 0
        #assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        if config['use_sideinfo']:
            sidevec = np.tile(atts, (batch_size, 1))
        else:
            sidevec = np.ndarray(shape=(batch_size, 0), dtype=np.float32)

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            buffer.append(data[data_index % len(data)])
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]

            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        return sidevec, batch, labels


