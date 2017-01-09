'''
generate latex tables of comparison of predictive pseudo-log-likelihood
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np 
import string
from scipy import sparse

import sys
sys.path.append('../util/')
from util import config_to_name

# step 0 parse configuration
#for K in [64, 128, 192, 256]:
for K in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:

    #dataset = 'restaurant'
    #config0 = dict(use_sideinfo=False, K=K, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=True)
    
    dataset = 'movie'
    fold = 0
    config0 = dict(use_sideinfo=False, fold=fold, K=K, max_iter=40000, reg_weight=10, exposure=False, cont_train=False, sample_ratio=0.1)
    config1 = config0.copy()
    config1.update({'exposure': True})

    config2 = config1.copy()
    config2.update({'use_sideinfo': True})
    
    data_path = '/rigel/dsi/users/ll3105/sa-data/' + dataset + '/'
    reviews = pickle.load(open(data_path + 'splits/test%d.pkl' % fold, 'rb'))
    
    losses = []
    for conf in [config0, config1, config2]:
        mfile = config_to_name(conf)
        #loss_array = pickle.load(open(data_path + 'splits/test/loss_' + mfile + '.pkl', "rb"))
        loader = pickle.load(open(data_path + 'splits/' + mfile + '.pkl', "rb"))
        loss_array = loader['loss_array']
        #rlosses = np.array([np.mean(rloss) for rloss in loss_array])

        for ir in xrange(len(loss_array)):
            _, ind, rate = sparse.find(reviews['scores'][ir, :])
            loss_array[ir] = loss_array[ir][ind]

        rlosses = np.concatenate(loss_array)
        losses.append(rlosses)
    
    print('\\multirow{2}{*}{K = %d}' % K, end='') 
    
    for i in xrange(len(losses)):
        print('& %.3f(%.3f)' % (np.mean(losses[i]), np.std(losses[i]) / np.sqrt(len(losses[i]))), end='')
    print('//')

    for i in xrange(len(losses)):
        if i == 0:
            print('& -', end='')
        else:
            diff = losses[i] - losses[0]
            print('& %.3f(%.3f)' % (np.mean(diff), np.std(diff) / np.sqrt(len(losses[i]))),end='')
    
    print('//')




