import numpy as np
import os
import cPickle as pkl
from scipy import sparse
import sys
sys.path.append('../')
from separate_sets import write_file_for_pemb

def save_data(savename, covname, obsmat, obscov, index):
    submat = obsmat[index, :]
    subcov = obscov[index, :]
    write_file_for_pemb(submat, savename)
    np.savetxt(covname, subcov, fmt='%.3f')

def stratify_data(obsmat, obscov, frac=[0.6, 0.1, 0.3], fold=0):

    n = obsmat.shape[0]
    ntrain = int(n * frac[0])
    nvalid = int(n * frac[1])
    ntest = int(n * frac[2])

    randperm = np.random.choice(n, size=n, replace=False)
    
    index = randperm[0:ntrain]
    savename = datapath + 'data_folds/%d/train.tsv' % fold
    covname = datapath + 'data_folds/%d/obscov_train.csv' % fold
    save_data(savename, covname, obsmat, obscov, index)
    
    index = randperm[ntrain:(ntrain + nvalid)]
    loadname = datapath + 'validation.tsv'
    savename = datapath + 'data_folds/%d/validation.tsv' % fold
    covname = datapath + 'data_folds/%d/obscov_valid.csv' % fold
    save_data(savename, covname, obsmat, obscov, index)

    index = randperm[(ntrain + nvalid):]
    savename = datapath + 'data_folds/%d/test.tsv' % fold
    covname = datapath + 'data_folds/%d/obscov_test.csv' % fold
    save_data(savename, covname, obsmat, obscov, index)
    

# load from pemb data
datapath = os.path.expanduser('~/storage/bird_data/movie/')
train = np.loadtxt(datapath + 'train.tsv')
test = np.loadtxt(datapath + 'test.tsv')
valid = np.loadtxt(datapath + 'validation.tsv')
data = np.r_[train, test, valid]
data = data[:, [0, 1, 3]]
print(data.shape)

# id to index
user_id = np.unique(data[:, 0])
item_id = np.unique(data[:, 1])
np.savetxt(datapath + 'user_id.csv', user_id, fmt='%d')
np.savetxt(datapath + 'item_id.csv', item_id, fmt='%d')

user_dict = dict(zip(*(user_id, np.arange(user_id.shape[0]))))
item_dict = dict(zip(*(item_id, np.arange(item_id.shape[0]))))

data[:, 0] = np.array([user_dict[iu] for iu in data[:, 0]])
data[:, 1] = np.array([item_dict[it] for it in data[:, 1]])
obsmat = sparse.coo_matrix((data[:, 2], (data[:, 0], data[:, 1]))).tocsr()
print(obsmat.shape)

# feature to obscov
feature = pkl.load(open(datapath + 'user_feature.pkl', 'rb'))
obscov = feature[user_id.astype(int) - 1, :]
print(obscov.shape)

for ifold in xrange(10):
    stratify_data(obsmat, obscov, frac=[0.6, 0.1, 0.3], fold=ifold)

