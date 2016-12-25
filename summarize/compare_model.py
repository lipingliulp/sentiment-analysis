from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np 
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
config = dict(use_sideinfo=False, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)
#config = dict(use_sideinfo=False, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True)

mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
emb_model = train['model']
config['num_neg'] = 5000
loss_array = evaluate_emb(testset, emb_model, config, voc_dict)
pickle.dump(loss_array, open('loss_exp.pkl', "wb"))

print('configuration is ', config)
print('loss mean and std are ', np.mean(loss_array), np.std(loss_array))


