from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np 
import re
import tensorflow as tf
import string

import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name
from emb_model import evaluate_emb 


# step 0 parse configuration
config = dict(use_sideinfo=False, K=64, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)

for iarg in xrange(1, len(sys.argv)):
    arg = sys.argv[iarg]
    key, strv = string.split(arg, '=')

    value = False if strv == 'False' else (True if strv == 'True' else int(strv)) 
    
    if key in config:
        config[key] = value
    else:
        raise Exception(key + ' is not a key of the config')

print('The configuration is: ')
print(config)

# Step 1: load data
dataset = 'restaurant'
data_path = '/rigel/dsi/users/ll3105/sa-data/' + dataset + '/'

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

mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
emb_model = train['model']
loss_array = evaluate_emb(testset, emb_model, config, voc_dict)
pickle.dump(loss_array, open(data_path + 'splits/test/loss_' + mfile, "wb"))

loss_array = np.concatenate(loss_array)
print('configuration is ', config)
print('loss mean and std are ', np.mean(loss_array), np.std(loss_array))



