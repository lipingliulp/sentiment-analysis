import tarfile
import json
import re
import cPickle as pickle
import sys
sys.path.insert(0, '../../util/')
from util import build_dictionary

data_path = '../../data/'
dataset = 'wikipedia'

files = ['tagged.en/englishEtiquetado_' + str(ind * 10000) + '_' + str(ind * 10000 + 10000) for ind in xrange(10)]

vocabulary_size = 20000

words = []

for filename in files:
    with open(data_path + dataset + '/' + filename, mode='r') as txtfile:

        while True:
            line = txtfile.readline()
            if line == '':
                break
            if line == '\n':
                continue

            tokens = line.split()
            words.append(tokens[0].lower())

print('Total number of words is %d' % len(words))

count, dictionary, reverse_dictionary = build_dictionary(words, vocabulary_size)
voc_dict = dict(dic=dictionary, rev_dic = reverse_dictionary, freq = count)
pickle.dump(voc_dict, open(data_path + dataset + '/voc_dict.pkl', 'wb'))



