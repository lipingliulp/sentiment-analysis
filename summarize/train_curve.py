from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

data_path = '../data/restaurant/'

# load model
config = dict(use_sideinfo=True, K=128, max_iter=300000, half_window=1, reg_weight=1.0, num_neg=50)
mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
logg_side = train['logg']
step, loss = zip(*logg_side)
print(len(step))
index = range(130, len(step) - 2)
plt.ylim(2.5, 4)
plt.xlim(index[0] * 2000, index[-1] * 2000)
line1, = plt.plot([step[i] for i in index], [loss[i] for i in index], color='red', label='w/ context attributes')

config = dict(use_sideinfo=False, K=128, max_iter=300000, half_window=1, reg_weight=1.0, num_neg=50)
mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
logg = train['logg']
step, loss = zip(*logg)
print(loss)
line2, = plt.plot([step[i] for i in index], [loss[i] for i in index], color='blue', label='w/o context attributes')

plt.legend()
plt.show()





