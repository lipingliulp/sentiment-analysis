import tarfile
import json
import re
import cPickle as pickle
import sys
sys.path.insert(0, '../util/')
from util import build_dictionary

data_path = '../data/'
dataset = 'restaurant'
tar_file = 'yelp_dataset_challenge_academic_dataset.tar'
tfile = tarfile.open(data_path + tar_file, mode='r')
rfile = tfile.extractfile('yelp_academic_dataset_review.json')
busi_dict = pickle.load(open(data_path + 'busi_dict.pkl', 'rb'))

vocabulary_size = 20000
num_review = 200000

btype = 'Restaurants'
att1= 'Price Range'
att2 = 'Outdoor Seating'
att3 = 'Waiter Service'
att4 = 'Ambience'

ireview = 0
words = list()
while True:
    line = rfile.readline()
    if line == '':
        break
    review = json.loads(line)
    
    business = busi_dict[review['business_id']]
    if not (btype in business['categories'] and 
             att1 in business['attributes'] and 
             att2 in business['attributes'] and 
             att3 in business['attributes'] and 
             att4 in business['attributes']):

        continue

    rwords = re.sub('[^A-Za-z0-9]+', ' ', review['text'].encode('ascii','ignore')).lower().split()
    if len(rwords) == 0:
        continue

    words.extend(rwords)

    ireview = ireview + 1

    if ireview % 1000 == 0:
        print('parsed review %d' % ireview)

    if ireview == num_review:
        break

print('Total number of words is %d' % len(words))

count, dictionary, reverse_dictionary = build_dictionary(words, vocabulary_size)
voc_dict = dict(dic=dictionary, rev_dic = reverse_dictionary, freq = count)
pickle.dump(voc_dict, open(data_path + dataset + '/voc_dict.pkl', 'wb'))



