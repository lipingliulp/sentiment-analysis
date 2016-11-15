import tarfile
import json
import re 
import numpy as np
import cPickle as pickle

data_path = '../data/'
dataset = 'restaurant'
tar_file = 'yelp_dataset_challenge_academic_dataset.tar'
tfile = tarfile.open(data_path + tar_file, mode='r')
rfile = tfile.extractfile('yelp_academic_dataset_review.json')

restaurant_dict = pickle.load(open(data_path + dataset + '/voc_dict.pkl', "rb"))
busi_dict = pickle.load(open(data_path + 'busi_dict.pkl', "rb"))
dictionary = restaurant_dict['dic']
reverse_dictionary = restaurant_dict['rev_dic'] 

num_review = 200000
btype = 'Restaurants'
att1= 'Price Range'
att2 = 'Outdoor Seating'
att3 = 'Waiter Service'
att4 = 'Ambience'
amb_list = ["romantic", "intimate", "classy", "hipster", "divey", "touristy", "trendy", "casual", "upscale"]

ireview = 0
instances = list()
while True:
    line = rfile.readline()
    if line == '':
        break
    review = json.loads(line)

    if ireview == 0:
        print review
    
    business = busi_dict[review['business_id']]
    if not (btype in business['categories'] and 
             att1 in business['attributes'] and 
             att2 in business['attributes'] and 
             att3 in business['attributes'] and 
             att4 in business['attributes']):
        continue

    rwords = re.sub('[^A-Za-z0-9]+', ' ', review['text'].encode('ascii','ignore')).lower().split()
    if len(rwords) < 10:
        continue

    iwords = [(dictionary[word] if word in dictionary else dictionary['UNK'])  for word in rwords]
    text = np.array(iwords)

    rtype = ['Food' in business['categories'],
             'Fast Food' in business['categories'], 
             'Bars' in business['categories'], 
             'Cafes' in business['categories'], 
             'Pizza' in business['categories'], 
             'Burgers' in business['categories'], 
             'Diners' in business['categories'], 
             'Mexican' in business['categories'], 
             'Italian' in business['categories'], 
             'Indian' in business['categories'], 
             'Thai' in business['categories'], 
             'American (Traditional)' in business['categories'], 
             'Chinese' in business['categories']] 
    
    price   = business['attributes'][att1]
    outseat = business['attributes'][att2]
    waiter  = business['attributes'][att3]
    stars   = review['stars']

    ambdict = business['attributes'][att4]
    amb     = [(ambdict[ambtype] if ambtype in ambdict else False) for ambtype in amb_list]

    atts = np.array(rtype + [price, outseat, waiter, stars] + amb, dtype=np.float32)
    instance = dict(text=text, atts=atts) 
    instances.append(instance)

    ireview = ireview + 1
    if ireview % 1000 == 0:
        print('Parsed %d reviews...' % ireview)
    if ireview == num_review:
        break

pickle.dump(instances, open(data_path + 'restaurant/reviews.pkl', 'wb'))






#reviews = list()
#
#for i in xrange(100000): 
#    rfile.readline():
#    review = json.loads(line)
#    rwords = re.sub('[^A-Za-z0-9]+', ' ', review['text'].encode('ascii','ignore')).lower().split()
#    if len(rwords) == 0:
#        continue
#    data = [(dictionary[word] if word in dictionary else dictionary['UNK'])  for word in rwords]
#
#
#
#    reviews.append(dict(data=data, stars=review['stars']))
#
#print('Overall %d reviews' % len(reviews))
#
#print('Sample data', reviews[0]['data'][:10], [reverse_dictionary[i] for i in reviews[0]['data'][:10]])



