import sys
sys.path.insert(0, '../util/')
from util import config_to_name
import numpy as np
from scipy.spatial import distance
import cPickle as pickle

def load_model(config, path):
    mfile = config_to_name(config) + '.pkl'
    loader = pickle.load(open(path + mfile, "rb"))
    model = loader['model']

    return model['alpha'], model['rho'], model['weight'], model['invmu']

def nearest_words(alpha, rand_ind):
    
    select = alpha[rand_ind, :]
    dist = distance.cdist(select, alpha)
    indices = np.argsort(dist, axis=-1)
    neighbors = indices[:, 0:9]

    return neighbors
    
def print_neighbors(iwords, neighbors):

    for i, iword in zip(range(len(iwords)), iwords):
        s = reverse_dictionary[iword] + ': '

        for nei in neighbors[i, :]:
            s = s + reverse_dictionary[nei] + ','

        print(s)

def top_words(score, num):
    ind = np.argsort(- score)
    words = [reverse_dictionary[ind[i]] for i in xrange(num)] 

    return words


dataset = 'restaurant'
data_path = '/rigel/dsi/users/ll3105/sa-data/' + dataset + '/'
voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
reverse_dictionary = voc_dict['rev_dic']
dictionary = voc_dict['dic']


config1 = dict(use_sideinfo=False, K=64, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=True)
alpha1, rho1, weight1, invmu1 = load_model(config1, data_path + 'splits/')

config2 = dict(use_sideinfo=False, K=64, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)
alpha2, rho2, weight2, invmu2 = load_model(config2, data_path + 'splits/')

config3 = dict(use_sideinfo=True, K=64, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=True, cont_train=True)
alpha3, rho3, weight3, invmu3 = load_model(config3, data_path + 'splits/')

#rand_ind = np.random.choice(alpha1.shape[0], 100, replace=False)
words = ['ramen', 'delicious', 'disgusting', 'expensive', 'noisy', 'waiter', 'dirty', 'sushi', 'pepperoni', 'water', 'restaurant', 
         'guest', 'pricy', 'coffee', 'early', 'carry', 'often', 'back', 'again', 'mcdonald']
rand_ind = np.array([dictionary[word] for word in words])

print('==================================================================================')
neighbors1 = nearest_words(alpha1, rand_ind)
neighbors2 = nearest_words(alpha2, rand_ind)
neighbors3 = nearest_words(alpha3, rand_ind)

nneq = np.sum(neighbors1 != neighbors2, axis=1) + np.sum(neighbors2 != neighbors3, axis=1) + np.sum(neighbors1 != neighbors3, axis=1)
top_diff = np.argsort(-nneq)[0 : 15]
#top_diff = range(0, 100)

print_neighbors(rand_ind[top_diff], neighbors1[top_diff, 1:])
print('-------------')
print_neighbors(rand_ind[top_diff], neighbors2[top_diff, 1:])
print('-------------')
print_neighbors(rand_ind[top_diff], neighbors3[top_diff, 1:])

print('==================================================================================')


def neighbor_rank(word_ref, words, alpha):

    w_ind = dictionary[word_ref]
    w_dist = distance.cdist(alpha[w_ind : w_ind + 1, :], alpha)
    sind = np.argsort(w_dist)

    ranks = []
    for word in words:
        r_ind = dictionary[word]
        ind = np.nonzero(sind == r_ind)[1][0]
        ranks.append(ind)

    return ranks 

word_ref = 'pizza'
words = ['crust', 'pizzas', 'pepperoni', 'wings', 'delivery', 'calzone', 'pie']
print(words)
nr = neighbor_rank(word_ref, words, alpha1)
print(nr)
nr = neighbor_rank(word_ref, words, alpha2)
print(nr)
nr = neighbor_rank(word_ref, words, alpha3)
print(nr)

word_ref = 'pizza'
words = ['sushi', 'breakfast', 'rice', 'pho', 'pork', 'egg', 'pancakes']
print(words)
nr = neighbor_rank(word_ref, words, alpha1)
print(nr)
nr = neighbor_rank(word_ref, words, alpha2)
print(nr)
nr = neighbor_rank(word_ref, words, alpha3)
print(nr)

print('==================================================================================')

print('"pizza" feature has the largest weight on word:')
print(top_words(weight3[:, 4], 8))
print('"pizza" feature has the smallest weight on word:')
print(top_words(-weight3[:, 4], 8))

print('"waiter" feature has the largest weight on word:')
print(top_words(weight3[:, 15], 8))
print('"waiter" feature has the smallest weight on word:')
print(top_words(-weight3[:, 15], 8))



print('"outseat" feature has the largest weight on word:')
print(top_words(weight3[:, 14], 8))
print('"outseat" feature has the smallest weight on word:')
print(top_words(-weight3[:, 14], 8))




print('==================================================================================')
pind = dictionary['beef']
eind = dictionary['juicy']
sind = dictionary['bread']

def word_relation(alpha, pind, eind, sind):
    target = alpha[sind, :] + (alpha[eind, :] - alpha[pind, :])
    dist = distance.cdist(target.reshape([1, -1]), alpha)
    iwords = np.argsort(dist[0, :])[0 : 10]
    candidates = [reverse_dictionary[iword] for iword in iwords]
    return candidates


candidates = word_relation(alpha1, pind, eind, sind) 
print('Words nearest to the target are ')
print(candidates)

candidates = word_relation(alpha2, pind, eind, sind)
print('Words nearest to the target are ')
print(candidates)



