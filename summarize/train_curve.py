from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

data_path = '/rigel/dsi/users/ll3105/sa-data/restaurant/'

K = 128 
# load model
config0 = dict(use_sideinfo=False, K=K, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=False)

config1 = config0.copy()
config1['cont_train'] = True
config1['max_iter'] = 800000

config2 = config1.copy() 
config2['exposure'] = True
config2['max_iter'] = 800000

config3 = config2.copy()
config3['use_sideinfo'] = True
config3['max_iter'] = 800000

logg = []
for config in [config0, config1, config2, config3]:
    mfile = config_to_name(config) + '.pkl'
    loader = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
    logg.append(loader['logg'])


pickle.dump(logg, open('lines%d.pkl' % K, 'wb'))



