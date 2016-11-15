from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import json
import re
import tensorflow as tf

import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name
from emb_model import evaluate_emb 

# Step 1: load data

dataset = 'restaurant'
data_path = '../data/' + dataset + '/'

# load dictionary
voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
dictionary = voc_dict['dic']
reverse_dictionary = voc_dict['rev_dic']
vocabulary_size = len(dictionary)
print('Vocabulary size is %d' % vocabulary_size)

# load data
tsfile = data_path + '/splits/test0.pkl'
testset = pickle.load(open(tsfile, 'rb'))
print('Overall %d reviews' % len(testset))

# load model
config = dict(use_sideinfo=True, K=128, max_iter=200000, 
              half_window=1, voc_size=len(dictionary), reg_weight=1.0, num_neg=12)

mfile = config_to_name(config) + '.pkl'
emb_model = pickle.load(open(data_path + 'splits/' + mfile, "rb"))

avg_loss = evaluate_emb(testset, emb_model, config)

print('configuration is ', config)
print('loss is ', avg_loss)


