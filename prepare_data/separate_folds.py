import cPickle as pickle
import numpy as np


data_path = '../data/'
dataset = 'restaurant'

review_set = pickle.load(open(data_path + dataset + '/' + 'reviews.pkl', 'rb'))

nins = len(review_set)
trindex = set(np.random.choice(nins, size=nins / 2, replace=False, p=None))

trainset = list()
testset = list()
for i in xrange(nins):
    if i in trindex:
        trainset.append(review_set[i])
    else:
        testset.append(review_set[i])

trfile = data_path + dataset + '/splits/train0.pkl'
pickle.dump(trainset, open(trfile, 'wb'))

tsfile = data_path + dataset + '/splits/test0.pkl'
pickle.dump(testset, open(tsfile, 'wb'))



