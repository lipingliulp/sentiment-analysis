
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import sys
sys.path.insert(0, '../util/')
sys.path.insert(0, '../word2vec/')
from util import config_to_name

from emb_model import generate_batch
from emb_model import fit_emb
import matplotlib.pyplot as plt

np.random.seed(seed=27)

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
config = dict(use_sideinfo=False, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=False)

print(config_to_name(config))

# Step 3: Function to generate a training batch for the skip-gram model.
sidevec, batch, labels = generate_batch(trainset[0], config=config)
#print(trainset[0])
for i in range(8):
    msg = ''
    for j in xrange(2 * config['half_window']):
        msg = msg + str(batch[i, j]) + ',' + reverse_dictionary[batch[i, j]] + ';'
    print(msg, ' *', sidevec[0:3],
          '->', labels[i], reverse_dictionary[labels[i]])

# Step 4: Fit a emb model
if config['cont_train']:
    dummy_config = config.copy()
    dummy_config['use_sideinfo'] = False 
    dummy_config['exposure'] = False 
    dummy_config['max_iter'] = 300000 
    mfile = config_to_name(dummy_config) + '.pkl'
    print(mfile) 
    train_noside = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
    init_model = train_noside['model']
else:
    init_model = None

emb_model, logg = fit_emb(trainset, config, voc_dict, init_model)

# Step 5: Save result and Visualize the embeddings.

mfile = config_to_name(config) + '.pkl'
pickle.dump(dict(model=emb_model, logg=logg), open(data_path + 'splits/' + mfile, "wb"))

print('Training done!')
