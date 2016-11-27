import tarfile
import cPickle as pickle
import json

data_path = '../data/'
tar_file = 'yelp_dataset_challenge_academic_dataset.tar'

tfile = tarfile.open(data_path + tar_file, mode='r')
bfile = tfile.extractfile('yelp_academic_dataset_business.json')


busi_dict = dict()
while True:
    line = bfile.readline()
    if line == '':
        break
    busi = json.loads(line)
    busi_dict[busi['business_id']] = busi

pickle.dump(busi_dict, open(data_path + 'dictionary/busi_dict.pkl', 'wb'))


