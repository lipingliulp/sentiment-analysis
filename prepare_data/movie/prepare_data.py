import numpy as np
import os
import cPickle as pkl
from scipy import sparse
import sys

def save_data(savename, obsmat, obscov, index):
    reviews = dict(scores=obsmat[index, :], atts=obscov[index, :])
    pkl.dump(reviews, open(savename, 'wb'))

def stratify_data(obsmat, obscov, frac=0.7, fold=0):

    n = obsmat.shape[0]
    ntrain = int(n * frac)
    ntest = n - ntrain 

    randperm = np.random.choice(n, size=n, replace=False)
    
    index = randperm[0:ntrain]
    savename = datapath + 'splits/train%d.pkl' % fold
    save_data(savename, obsmat, obscov, index)
    
    index = randperm[ntrain:]
    savename = datapath + 'splits/test%d.pkl' % fold
    save_data(savename, obsmat, obscov, index)
    

# load from pemb data
datapath = os.path.expanduser('../../data/movie/')
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
    stratify_data(obsmat, obscov, frac=0.7, fold=ifold)

