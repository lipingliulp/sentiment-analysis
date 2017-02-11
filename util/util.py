import collections
import numpy as np
import matplotlib.pyplot as plt

def build_dictionary(words, vocabulary_size):
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    print('Overall vocabulary size is ', len(counter))
    count.extend(counter.most_common(vocabulary_size - 1))
    del counter

    dictionary = dict()
    numsum = 0 
    for word, num in count:
          dictionary[word] = len(dictionary)
          numsum = numsum + num

    count[0][1] = len(words) - numsum
    print count[0]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary

def config_to_name(config, fixorder=True):
    keys = config.keys()
    keys.sort()
    name = ''
    for key in keys:
        name = name + '.' + key + '=' + str(config[key])
    name = name[1:]
    return name 


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(38, 38))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def npgammaln(x):
    # fast approximate gammaln from paul mineiro
    # http://www.machinedlearnings.com/2011/06/faster-lda.html
    logterm = np.log (x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * np.log (xp3)




