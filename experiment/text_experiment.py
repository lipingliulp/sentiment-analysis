
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

from text_model import generate_batch
from text_model import fit_emb
from text_model import evaluate_emb

# Step 1: load data

def text_experiment(config):

    np.random.seed(seed=27)
    
    dataset = 'wikipedia'
    data_path = '../data/' + dataset + '/'
    
    trfile = data_path + '/splits/train%d.pkl' % config['fold']
    trainset = pickle.load(open(trfile, 'rb'))
    print('Overall %d reviews' % len(trainset))
    
    # Step 2: Fit a emb model
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
    
    tag_dict = pickle.load(open(data_path + 'simple_tag_dict.pkl', 'rb'))
    tags = [''] * 12
    for tag in tag_dict.keys():
        tags[tag_dict[tag]] = tag

    voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
    dictionary = voc_dict['dic']
    reverse_dictionary = voc_dict['rev_dic']
    sample_doc = trainset[16][0:20, :]
    print('Sample data', sample_doc[:, 0], [reverse_dictionary[i] for i in sample_doc[:, 0]])
    print('Prior tags', sample_doc[:, 1], [tags[i] for i in sample_doc[:, 1]])

    emb_model, logg = fit_emb(trainset, config, init_model)
    print('Training done!')
    
    tsfile = data_path + '/splits/test%d.pkl' % config['fold']
    testset = pickle.load(open(tsfile, 'rb'))
    print('Overall %d reviews' % len(testset))
    
    loss_array, pos_loss_array = evaluate_emb(testset, emb_model, config)
    # Step 4: Save result and Visualize the embeddings.
    mfile = config_to_name(config) + '.pkl'
    pickle.dump(dict(model=emb_model, logg=logg, loss_array=loss_array, pos_loss_array=pos_loss_array), open(data_path + 'splits/' + mfile, "wb"))




