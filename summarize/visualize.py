from sklearn.manifold import TSNE
import sys
sys.path.insert(0, '../util/')
from util import plot_with_labels
from util import config_to_name
import cPickle as pickle
import pandas as pd
import numpy as np

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only =500 

# for text data
data_path = '../data/wikipedia/'
voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
reverse_dictionary = voc_dict['rev_dic']
#config = dict(use_sideinfo=False, K=128, max_iter=200000, half_window=1, reg_weight=1.0, num_neg=10)
#config = dict(use_sideinfo=True, K=32, max_iter=40000, ar_sigma2=1, w_sigma2=1, exposure=True, cont_train=False, sample_ratio=0.01, 
#              dist='bernoulli', fold=0)

config = dict(use_sideinfo=False, K=32, max_iter=100000, ar_sigma2=1, w_sigma2=1, exposure=False, cont_train=False, sample_ratio=0.01, 
              dist='bernoulli', fold=0)

for i in xrange(len(reverse_dictionary)):
    reverse_dictionary[i] = reverse_dictionary[i].decode('utf8', "ignore").encode("ascii","ignore")
    

## for movie data
#data_path = '../data/movie/'
#items = pd.read_csv(data_path + 'u.item', sep='|', header=None)
#names = items[1]
#item_id = np.loadtxt(data_path + 'item_id.csv', dtype=int)
#reverse_dictionary = [names[i] for i in item_id]
#print(reverse_dictionary)

#config = dict(use_sideinfo=True, K=8, max_iter=20000, reg_weight=0.01, exposure=True, cont_train=False, sample_ratio=0.1, fold=0)

mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
alpha = train['model']['alpha']

low_dim_embs = tsne.fit_transform(alpha)
labels = reverse_dictionary
index = np.random.choice(len(labels), size=plot_only, replace=False)
low_dim_embs = low_dim_embs[index, :]
labels = [labels[i] for i in index]
plot_with_labels(low_dim_embs, labels)



