import numpy as np
import os
import cPickle as pkl

def prepare_data(loadname, savename, covname, user_dict, item_dict, feature):
    ar = np.loadtxt(loadname)

    user_id = ar[:, 0]
    assert(np.sum(user_id - user_id.astype(np.int32)) == 0)
    index = user_id.astype(np.int32) - 1
    obscov = feature[index, :] 
    np.savetxt(covname, obscov, fmt='%.3f')

    ar[:, 0] = np.array([user_dict[iu] for iu in ar[:, 0]])
    ar[:, 1] = np.array([item_dict[it] for it in ar[:, 1]])

    index = [2, 1, 0, 3]
    ar = ar[:, index]    

    np.savetxt(savename, ar, fmt='%d', delimiter='\t')


datapath = os.path.expanduser('~/storage/bird_data/movie/')

loadname = datapath + 'train.tsv'
savename = datapath + 'data_folds/0/train.tsv'
covname = datapath + 'data_folds/0/obscov_train.csv'

traindata = np.loadtxt(loadname)
user_id = np.unique(traindata[:, 0])
item_id = np.unique(traindata[:, 1])
np.savetxt(datapath + 'user_id.csv', user_id, fmt='%d')
np.savetxt(datapath + 'item_id.csv', item_id, fmt='%d')

user_dict = dict(zip(*(user_id, np.arange(user_id.shape[0]))))
item_dict = dict(zip(*(item_id, np.arange(item_id.shape[0]))))
feature = pkl.load(open(datapath + 'user_feature.pkl', 'rb'))

prepare_data(loadname, savename, covname, user_dict, item_dict, feature)

loadname = datapath + 'test.tsv'
savename = datapath + 'data_folds/0/test.tsv'
covname = datapath + 'data_folds/0/obscov_test.csv'
prepare_data(loadname, savename, covname, user_dict, item_dict, feature)

loadname = datapath + 'validation.tsv'
savename = datapath + 'data_folds/0/validation.tsv'
covname = datapath + 'data_folds/0/obscov_valid.csv'
prepare_data(loadname, savename, covname, user_dict, item_dict, feature)


