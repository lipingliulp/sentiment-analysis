import cPickle as pickle
import numpy as np

data_path = '../../data/'
dataset = 'wikipedia'

docs = pickle.load(open(data_path + dataset + '/text.pkl', 'rb'))
flags = np.random.rand(len(docs)) > 0.33

train = []
test = []

for i, item in enumerate(docs):
    if docs[i].shape[0] < 3: 
        print('document too short, shape is ')
        print(docs[i].shape)
        continue
    
    if flags[i]:
        train.append(docs[i])
    else:
        test.append(docs[i])

print('%d for training and %d for testing' % (len(train), len(test)))

pickle.dump(train, open(data_path + dataset + '/splits/train0.pkl', 'wb'))
pickle.dump(test,  open(data_path + dataset + '/splits/test0.pkl', 'wb'))



