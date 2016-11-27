from sklearn.manifold import TSNE
import sys
sys.path.insert(0, '../util/')
from util import plot_with_labels
from util import config_to_name
import cPickle as pickle

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500

data_path = '../data/restaurant/'
voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
reverse_dictionary = voc_dict['rev_dic']

config = dict(use_sideinfo=False, K=128, max_iter=200000, half_window=1, reg_weight=1.0, num_neg=10)

mfile = config_to_name(config) + '.pkl'
train = pickle.load(open(data_path + 'splits/' + mfile, "rb"))
alpha = train['model']['alpha']

low_dim_embs = tsne.fit_transform(alpha[:plot_only,:])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels)













