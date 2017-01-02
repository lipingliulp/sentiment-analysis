import numpy as np
import os
import scipy.sparse as sparse
import cPickle as pkl

dataset = 'movie'
data_path = '~/storage/sa-data/' + dataset + '/'
data_path = os.path.expanduser(data_path)

dfile = data_path + 'u.data'
data = np.loadtxt(dfile)

data = data[:, :3]
data[:, 2] = data[:, 2] - 2
data[data[:, 2] < 0, 2] = 0
data[:, 2] = data[:, 2] + 1



#order = np.argsort(data[:, 0])
#data = data[order, :]
data = sparse.coo_matrix((data[:, 2], (data[:, 0] - 1, data[:, 1] - 1)), dtype=np.int16).toarray()

for i in xrange(20): 
    rowsum = np.sum(data > 0, 0)
    colsum = np.sum(data > 0, 1)

    m_flag = rowsum > 20 
    u_flag = colsum > 20 
    
    print(np.sum(m_flag))
    print(np.sum(u_flag))

    if np.sum(m_flag) == data.shape[1] and np.sum(u_flag) == data.shape[0]:
        break

    data = data[u_flag.nonzero()[0][:, np.newaxis], m_flag]

    print('' + str(i) + str(data.shape))

    
train_frac = 0.67

un = np.random.rand(data.shape[0])
tind = un < train_frac
sind = np.logical_not(tind) 

train = data[tind, :]
test = data[sind, :]

pkl.dump(train, open(data_path + 'train0.pkl', 'wb'))
pkl.dump(test, open(data_path + 'test0.pkl', 'wb'))

