import numpy as np
import os
import cPickle as pkl
from scipy import sparse
import sys
import pandas 

def save_data(savename, obsmat, obscov, index):
    reviews = dict(scores=obsmat[index, :], atts=obscov[index, :])
    pkl.dump(reviews, open(savename, 'wb'))

# load from pemb data
datapath = os.path.expanduser('~/storage/sa-data/market/')
data = np.loadtxt(datapath + 'data_iri_sessions12_poisson.txt', delimiter=',')

# replace item id with item index
smat = data[:, [0, 2, 3]]
item_id = np.unique(smat[:, 1])
np.savetxt(datapath + 'item_id.csv', item_id, fmt='%d')
item_dict = dict(zip(*(item_id, np.arange(item_id.shape[0]))))
for i in xrange(smat.shape[0]):
    smat[i, 1] = item_dict[smat[i, 1]]

obsmat = sparse.coo_matrix((smat[:, 2], (smat[:, 0], smat[:, 1]))).tocsr()
print(obsmat.shape)

# find feature vectors of sessions
su_pairs = data[:, [0, 1]].astype(int)
# unique rows 
dummy = np.ascontiguousarray(su_pairs).view(np.dtype((np.void, su_pairs.dtype.itemsize * su_pairs.shape[1])))
_, idx = np.unique(dummy, return_index=True)
su_pairs = su_pairs[idx]
np.savetxt(datapath + 'session_user.csv', su_pairs, fmt='%d')

#feature = np.loadtxt(datapath + 'user_feature.csv', delimiter=',', skiprows=1)
feature = pandas.read_csv(datapath + 'user_feature.csv', delimiter=',').as_matrix()
uind_dict = dict(zip(*(feature[:, 0].astype(int), np.arange(feature.shape[0]))))

obscov = np.zeros((su_pairs.shape[0], feature.shape[1] - 1))
for i in xrange(su_pairs.shape[0]):
    si = su_pairs[i, 0]
    ui = su_pairs[i, 1]
    obscov[si, :] = feature[uind_dict[ui], 1:]

print(obscov.shape)

nfold = 3 
rind = np.random.permutation(obsmat.shape[0]) % nfold

for ifold in xrange(nfold):
    
    savename = datapath + 'splits/train%d.pkl' % ifold
    save_data(savename, obsmat, obscov, rind != ifold)
    
    savename = datapath + 'splits/test%d.pkl' % ifold
    save_data(savename, obsmat, obscov, rind == ifold)
 
