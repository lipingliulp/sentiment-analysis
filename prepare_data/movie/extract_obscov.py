import pandas as pd
import os
import numpy as np
import cPickle as pkl

datapath = os.path.expanduser('~/storage/bird_data/movie/') 
user = pd.read_csv(datapath + 'u.user', sep='|', names=['id', 'age', 'gender', 'prof', 'zip'])

age = np.array(user['age'])
age = age / float(np.max(age))

gender = np.array(user['gender'] == 'F', dtype=int)

uprof = np.unique(user['prof'])
pdict = dict(zip(*(uprof, range(len(uprof)))))
prof = np.array([pdict[ip] for ip in user['prof']])

pmat = np.zeros([user.shape[0], len(pdict)]) 
pmat[np.arange(user.shape[0]), prof] = 1

feature = np.c_[age, gender, pmat]
pkl.dump(feature, open(datapath + 'user_feature.pkl', 'wb'))



