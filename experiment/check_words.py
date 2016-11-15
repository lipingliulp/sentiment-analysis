from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import json
import re
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name
from emb_model import generate_batch

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
config = dict(use_sideinfo=False, K=128, max_iter=200000, context_type='skip_gram', skip_window=1, num_skips=2, voc_size=len(dictionary), reg=1.0)
mfile = config_to_name(config) + '.pkl'
emb_model = pickle.load(open(data_path + 'splits/' + mfile, "rb"))

# batch_size becomes a variable 


print(emb_model['rho'].shape)
print(emb_model['alpha'].shape)

basedim = emb_model['alpha'].shape[1]

for idim in xrange(basedim, emb_model['rho'].shape[1]):
    print('-----------------------------------------------------------')
    indword = np.argsort(- np.abs(emb_model['rho'][:, idim]))[0:5]
    print('Most correlated words to feature %d  are %s' % (idim, str([reverse_dictionary[iword] for iword in indword])))
    print(emb_model['rho'][indword, idim])
    
    randind = np.random.choice(np.arange(0, len(dictionary)), 6, replace=False)
    print('randomly chosen words:')
    print(reverse_dictionary[ind] for ind in randind)
    print(emb_model['rho'][randind, idim])


alpha_norm = np.sqrt(np.sum(np.square(emb_model['alpha']), axis=1))
emb_model['alpha'] = emb_model['alpha'] / alpha_norm[:, None]

sample_words = ['spaghetti', 'noodle', 'pad', 'pepperoni', 'burger', 'beef', 'sushi', 'service', 'happy', 'panda', 'mcdonald'] 

for word in sample_words:
    
    word_ind = dictionary[word]
    vec = emb_model['alpha'][word_ind, :]
    values = emb_model['alpha'].dot(vec)
    indw = np.argsort(- values)[1:10]

    print(word + ' is nearest to ' + str([reverse_dictionary[ind] for ind in indw]))
    

#print('Vector of grinder is ' + str(emb_model['alpha'][dictionary['grinder'], :]))
#print('Vector of yakiniku is ' + str(emb_model['alpha'][dictionary['yakiniku'], :]))
#print('Vector of sammys is ' + str(emb_model['alpha'][dictionary['sammys'], :]))


