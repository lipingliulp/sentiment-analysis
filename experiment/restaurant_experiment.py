
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


# Step 1: load data

def restaurant_experiment(config):

    np.random.seed(seed=27)
    
    dataset = 'restaurant'
    data_path = '/rigel/dsi/users/ll3105/sa-data/' + dataset + '/'
    
    trfile = data_path + '/splits/train0.pkl'
    trainset = pickle.load(open(trfile, 'rb'))
    print('Overall %d reviews' % len(trainset))
    
    voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
    dictionary = voc_dict['dic']
    reverse_dictionary = voc_dict['rev_dic']
    print('Sample data', trainset[0]['text'][:10], [reverse_dictionary[i] for i in trainset[0]['text'][:10]])
    
    # Step 2: Function to generate a training batch for the skip-gram model.
    sidevec, batch, labels = generate_batch(trainset[0], config=config)
    #print(trainset[0])
    for i in range(8):
        msg = ''
        for j in xrange(2 * config['half_window']):
            msg = msg + str(batch[i, j]) + ',' + reverse_dictionary[batch[i, j]] + ';'
        print(msg, ' *', sidevec[0:3],
              '->', labels[i], reverse_dictionary[labels[i]])
    
    # Step 3: Fit a emb model
    print(config_to_name(config))
    if config['cont_train']:
        dummy_config = config.copy()
        dummy_config['cont_train'] = False 
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
    
    # Step 4: Save result and Visualize the embeddings.
    mfile = config_to_name(config) + '.pkl'
    pickle.dump(dict(model=emb_model, logg=logg), open(data_path + 'splits/' + mfile, "wb"))
    
    print('Training done!')


