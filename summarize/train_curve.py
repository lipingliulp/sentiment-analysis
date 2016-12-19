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

data_path = '../data/restaurant/'

# load model
config0 = dict(use_sideinfo=False, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True)
config1 = dict(use_sideinfo=False, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)
config2 = dict(use_sideinfo=True, K=64, max_iter=300000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)

mfile = config_to_name(config0) + '.pkl'
loader = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
logg_exp = loader['logg']

mfile = config_to_name(config1) + '.pkl'
loader = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
logg_exp1 = loader['logg']

mfile = config_to_name(config2) + '.pkl'
loader = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
logg_side = loader['logg']

step_side = range(302000, 600000, 2000) + [600000]
loss_side = []
loss_exp1 = []
for i in xrange(1, 151):
    loss_side.append(sum(logg_side[(i - 1) * 2000 : i * 2000]) / 2000)
    loss_exp1.append(sum(logg_exp1[(i - 1) * 2000 : i * 2000]) / 2000)

step_exp, loss_exp = zip(*logg_exp)
step_exp = list(step_exp)
loss_exp = list(loss_exp)
step_exp[-1] = 300000

step = step_exp + step_side
loss = loss_exp + loss_exp1


font = {'size' : 28}
matplotlib.rc('font', **font)

plt.ylim(5, 8)
plt.xlim(0, 600000)
index = range(0, 301)
line1, = plt.plot(step, loss, color='red', label='w/o feats')
line2, = plt.plot(step_side, loss_side, color='blue', label='w/ feats')

plt.title('Negative log-likelihood of exposure model with and without features')
plt.ylabel('negative log-likelihood')
plt.xlabel('training mini-batches')
plt.legend()
plt.show()





