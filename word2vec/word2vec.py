
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import cPickle as pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import sys
sys.path.insert(0, '../util/')
from util import config_to_name

from emb_model import generate_batch
from emb_model import fit_emb
import matplotlib.pyplot as plt


# Step 1: load data

dataset = 'restaurant'
data_path = '../data/' + dataset + '/'

trfile = data_path + '/splits/train0.pkl'
trainset = pickle.load(open(trfile, 'rb'))
print('Overall %d reviews' % len(trainset))

voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
dictionary = voc_dict['dic']
reverse_dictionary = voc_dict['rev_dic']
print('Sample data', trainset[0]['text'][:10], [reverse_dictionary[i] for i in trainset[0]['text'][:10]])

# Step 2: set parameters of the model
config = dict(use_sideinfo=True, K=128, max_iter=300000, half_window=1, reg_weight=1.0, num_neg=10)

print(config_to_name(config))

# Step 3: Function to generate a training batch for the skip-gram model.
sidevec, batch, labels = generate_batch(trainset[0], config=config)
#print(trainset[0])
for i in range(8):
    msg = ''
    for j in xrange(2 * config['half_window']):
        msg = msg + str(batch[i, j]) + ',' + reverse_dictionary[batch[i, j]] + ';'
    print(msg, ' *', sidevec[i, 1:3],
          '->', labels[i], reverse_dictionary[labels[i]])

# Step 4: Fit a emb model
dummy_config = config.copy()
dummy_config['use_sideinfo'] = False
mfile = config_to_name(dummy_config) + '.pkl'
#init_model = pickle.load(open(data_path + 'splits/' + mfile, "rb"))

emb_model, logg = fit_emb(trainset, config, voc_dict, None)

# Step 5: Save result and Visualize the embeddings.

mfile = config_to_name(config) + '.pkl'
pickle.dump(dict(model=emb_model, logg=logg), open(data_path + 'splits/' + mfile, "wb"))

# plot curve

steps, avg_loss = zip(*logg)
plt.plot(steps, avg_loss)
plt.ylabel('average loss')
plt.ylabel('iterations')
plt.show()
