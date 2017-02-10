import tarfile
import json
import re 
import numpy as np
import cPickle as pickle

data_path = '../../data/'
dataset = 'wikipedia'
files = ['tagged.en/englishEtiquetado_' + str(ind * 10000) + '_' + str(ind * 10000 + 10000) for ind in xrange(0, 100)]

wiki_dict = pickle.load(open(data_path + dataset + '/voc_dict.pkl', "rb"))
dictionary = wiki_dict['dic']
reverse_dictionary = wiki_dict['rev_dic'] 

from freeling_tags_dict import reducing_dict
# read in reducing_dict
reduced_tags = list(set(reducing_dict.values()))

tag_dict = reducing_dict.copy()
simple_tag_dict = dict()
for tag in tag_dict.keys():
    rtag = reducing_dict[tag]
    itag = reduced_tags.index(rtag)
    tag_dict[tag] = itag  
    simple_tag_dict.update({rtag: itag})

ipunc = simple_tag_dict['punctuation']

print(tag_dict)
print(simple_tag_dict)
pickle.dump(simple_tag_dict, open(data_path + dataset + '/simple_tag_dict.pkl', 'wb'))


key_words = ['computer', 'programming', 'software', 'hardware', 'cpu', 'compiler', 'algorithm', 'statistics']
def line_to_pair(tokens):
    word = tokens[0].lower()
    tag = tokens[2]
    
    return word, tag

docs = []
doc = []
for filename in files:
    print('number of docs %d' % len(docs))
    print('processing file ' + filename + '...')

    with open(data_path + dataset + '/' + filename, mode='r') as txtfile:
        
        while True:
            line = txtfile.readline()
            if line == '':
                # end of the file 
                break

            if line.startswith('<doc '): # a new document
                doc_id = line
                # read a single doc
                doc = [] 
                while True:
                    line = txtfile.readline()
                    if line == '\n':
                        continue
                    if line.startswith('</doc>') or line == 'f':
                        # end of a document
                        break

                    tokens = line.split()
                    if len(tokens) < 3: # the bad line at the end of file
                        break

                    (word, tag) = line_to_pair(tokens)
                    if not (word in ['endofarticle']): # remove some bad words
                        doc.append((word, tag))
                
                if len(doc) <= 2: # bad doc
                    print('short doc' + doc_id)
                    print(doc)
                    continue 
                
                flag_cs = False
                for (word, tag) in doc:
                    if word in key_words:
                        flag_cs = True
                        break
                if flag_cs:
                    adoc = np.zeros((len(doc), 2), dtype=np.int32)
                    for i, (word, tag) in enumerate(doc):
                        if word in dictionary:
                            iword = dictionary[word]
                        else:
                            iword = 0
                        itag = tag_dict[tag] 
                        adoc[i, 0] = iword
                        adoc[i, 1] = itag

                    pflag = adoc[:, 0] != ipunc # remove punctuations
                    adoc = adoc[pflag, :]
                    docs.append(adoc)
                else:
                    pass
            else:
                print('bad line ' + line)
                raise Exception


print('number of documents is %d' % len(docs))

pickle.dump(tag_dict, open(data_path + dataset + '/tag_dict.pkl', 'wb'))
pickle.dump(docs, open(data_path + dataset + '/text.pkl', 'wb'))



