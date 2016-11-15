from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import json
import re
import tensorflow as tf

import sys
sys.path.append('../util/')
sys.path.append('../word2vec/')
from util import config_to_name
from emb_model import generate_batch

# Step 1: load data

dataset = 'restaurant'
data_path = '../data/' + dataset + '/'

# load dictionary
voc_dict = pickle.load(open(data_path + 'voc_dict.pkl', 'rb'))
dictionary = voc_dict['dic']
reverse_dictionary = voc_dict['rev_dic']
vocabulary_size = len(dictionary)
print('Vocabulary size is %d' % vocabulary_size)

# load data
tsfile = data_path + '/splits/test0.pkl'
testset = pickle.load(open(tsfile, 'rb'))
print('Overall %d reviews' % len(testset))

# load model
config = dict(use_sideinfo=True, K=128, max_iter=200000, context_type='skip_gram', skip_window=1, num_skips=2, voc_size=len(dictionary), reg=1.0)
mfile = config_to_name(config) + '.pkl'
emb_model = pickle.load(open(data_path + 'splits/' + mfile, "rb"))

# batch_size becomes a variable 
num_sampled = 64    # Number of negative examples to sample.


reviews = testset
num_review = len(reviews)
print('Sample data', reviews[0]['text'][:10], [reverse_dictionary[i] for i in reviews[0]['text'][:10]])

# Step 3: Function to generate a training batch for the skip-gram model.
graph = tf.Graph()
with graph.as_default():

  # Input data.
    if config['use_sideinfo']:
        train_sidevec = tf.placeholder(tf.float32, shape=[None, len(reviews[0]['atts'])])
    else:
        train_sidevec = tf.placeholder(tf.float32, shape=[None, 0])

    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None, 1])

  # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.constant(emb_model['alpha'])
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        embed = tf.concat(concat_dim=1, values=[embed, train_sidevec])

        # Construct the variables for the NCE loss
        nce_weights = tf.constant(emb_model['rho'])
        nce_biases = tf.constant(emb_model['b'])

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_sum(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

    # Compute the cosine similarity between minibatch examples and all embeddings.
    init = tf.initialize_all_variables()


# calculate errors on test set

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    sum_loss = 0
    num_pair = 0
    for ind, review in zip(range(len(reviews)), reviews):
        batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
        num_pair = num_pair + batch_labels.shape[0]
        feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}
        loss_val = session.run(loss, feed_dict=feed_dict)
        sum_loss += loss_val
        
        if (ind + 1) % 1000 == 0:
            print('Average loss at iteration %d is %f' % (ind, sum_loss / num_pair))

avg_loss = sum_loss / num_pair
  
print('configuration is ', config)
print('loss is ', avg_loss)


