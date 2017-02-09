import tarfile
import json
import re 
import numpy as np
import cPickle as pickle

data_path = '../../data/'
dataset = 'wikipedia'
files = ['tagged.en/englishEtiquetado_' + str(ind * 10000) + '_' + str(ind * 10000 + 10000) for ind in xrange(10)]

wiki_dict = pickle.load(open(data_path + dataset + '/voc_dict.pkl', "rb"))
dictionary = wiki_dict['dic']
reverse_dictionary = wiki_dict['rev_dic'] 

from freeling_tags_dict import reducing_dict
# read in reducing_dict
reduced_tags = list(set(reducing_dict.values()))

tag_dict = reducing_dict.copy()
for tag in tag_dict.keys():
    rtag = reducing_dict[tag]
    tag_dict[tag] = reduced_tags.index(rtag) 

print(tag_dict)
counter = np.zeros([len(tag_dict), len(tag_dict)])

itag0 = -1
itag1 = -1
itag2 = -1

docs = []
doc = []
for filename in files:
    with open(data_path + dataset + '/' + filename, mode='r') as txtfile:
        while True:
            line = txtfile.readline()
            if line == '':
                # end of the file 
                break
            if line == '\n':
                # end of a sentence
                itag0 = -1
                itag1 = -1
                itag2 = -1
                continue

            if line.startswith('</doc>') or line.startswith('<doc ') or line == 'f':
                # start of a new document. reset the recorder of the document
                if len(doc) > 0:
                    docs.append(np.array(doc))
                    doc = [] 
                continue
            
            tokens = line.split()

            word = tokens[0].lower()
            if word in dictionary:
                iword = dictionary[word]
            else:
                iword = 0

            tag = tokens[2]
            itag = tag_dict[tag] 

            doc.append([iword, itag])

            itag0 = itag1
            itag1 = itag2
            itag2 = itag
            if itag0 > 0 and itag2 > 0:
                counter[itag0, itag2] = counter[itag0, itag2] + 1

if len(doc) > 0:
    # record the last document if there is one
    docs.append(np.array(doc))
    doc = []

#np.savetxt('pair_cooccur.csv', counter)

print('number of documents is %d' % len(docs))

pickle.dump(tag_dict, open(data_path + dataset + '/tag_dict.pkl', 'wb'))
pickle.dump(docs, open(data_path + dataset + '/text.pkl', 'wb'))



