import numpy as np
import os
import scipy.sparse as sparse
import cPickle as pkl

dataset = 'movie'
data_path = '../../data/' + dataset + '/'
data_path = os.path.expanduser(data_path)

train = np.loadtxt(data_path + 'train.tsv')
valid = np.loadtxt(data_path + 'validation.tsv')
test = np.loadtxt(data_path + 'test.tsv')

train = np.r_[train, valid]
mat_t = sparse.coo_matrix((train[:, 3], (train[:, 0], train[:, 1])), dtype=np.int16).toarray()
print(np.unique(train[:, 0]).shape)
print(mat_t.shape)

mat_s = sparse.coo_matrix((test[:, 3], (test[:, 0], test[:, 1])), dtype=np.int16).toarray()
print(np.unique(test[:, 0]).shape)
#pkl.dump(mat_t, open(data_path + 'train0.pkl', 'wb'))
#pkl.dump(test, open(data_path + 'test0.pkl', 'wb'))

