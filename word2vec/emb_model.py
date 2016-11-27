from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
import random

import sys


# config = dict(K=K, voc_size=voc_size, use_sideinfo=use_sideinfo, half_window=half_window)
def fit_emb(reviews, config, voc_dict, init_model):

    embedding_size = config['K']
    vocabulary_size = len(voc_dict['dic']) 
    reverse_dictionary = voc_dict['rev_dic']
    wcount = [voc_dict['freq'][i][1] for i in xrange(vocabulary_size)]
    log_wcount = np.array(wcount, dtype=np.float32)
    log_wcount = 0.75 * np.log(log_wcount) 


    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_examples = np.random.choice(range(100), valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        if config['use_sideinfo']:
            dim_atts = len(reviews[0]['atts']) 
            train_sidevec = tf.placeholder(tf.float32, shape=[None, dim_atts], name='atts')
        else:
            train_sidevec = tf.placeholder(tf.float32, shape=[None, 0], name='atts')

        train_inputs = tf.placeholder(tf.int32, shape=[None, 2 * config['half_window']], name='train_contexts')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        if init_model == None:
            alpha = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            rho = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size + int(train_sidevec.get_shape()[1])],
                                stddev=1.0))
            bias = tf.Variable(tf.zeros([vocabulary_size]))
        else:
            alpha = tf.Variable(init_model['alpha'])
            bias = tf.Variable(init_model['b'])

            num_col = embedding_size + int(train_sidevec.get_shape()[1])
            ps = init_model['rho'].shape
            if num_col > ps[1]:
                init_rho = np.c_[init_model['rho'], np.zeros((ps[0], num_col - ps[1]))]
                rho = tf.Variable(init_rho, dtype=tf.float32)
            else: 
                rho = tf.Variable(init_model['rho'][:, :num_col], dtype=tf.float32)

        objective, loss = construct_graph(alpha, rho, bias, train_sidevec, train_inputs, train_labels, config, log_wcount)
        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(objective)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(alpha), 1, keep_dims=True))
        normalized_embeddings = alpha / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()


    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        loss_logg = []
        average_loss = 0
        loss_count = 0 
        for step in xrange(config['max_iter']):
            review = reviews[step % len(reviews)]
            batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
            feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
                
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            loss_count = loss_count + 1

            if step % 2000 == 0:
                if step > 0:
                  average_loss /= loss_count
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                loss_logg.append((step, average_loss))
                average_loss = 0
                loss_count = 0
    
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 20000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
   
         
        # logging the last result
        loss_logg.append((step, average_loss / loss_count))
        model = dict(alpha=alpha.eval(), rho=rho.eval(), b=bias.eval())

        return model, loss_logg



def evaluate_emb(reviews, model, config, voc_dict):

    embedding_size = config['K']
    vocabulary_size = len(voc_dict['dic'])
    wcount = [voc_dict['freq'][i][1] for i in xrange(vocabulary_size)]
    log_wcount = np.array(wcount, dtype=np.float32)
    log_wcount = 0.75 * np.log(log_wcount) 

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        if config['use_sideinfo']:
            dim_atts = len(reviews[0]['atts']) 
            train_sidevec = tf.placeholder(tf.float32, shape=[None, dim_atts], name='atts')
        else:
            train_sidevec = tf.placeholder(tf.float32, shape=[None, 0], name='atts')

        train_inputs = tf.placeholder(tf.int32, shape=[None, 2 * config['half_window']], name='train_contexts')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')

        # Ops and variables pinned to the CPU because of missing GPU implementation
        alpha = tf.constant(model['alpha'])
        rho = tf.constant(model['rho'])
        bias = tf.constant(model['b'])

        objective, loss = construct_graph(alpha, rho, bias, train_sidevec, train_inputs, train_labels, config, log_wcount)

        # Add variable initializer.
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        loss_sum = 0
        num_pairs = 0
        for step in xrange(len(reviews)):
            review = reviews[step % len(reviews)]
            batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
            feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}
    
            loss_val = session.run(loss, feed_dict=feed_dict)
            loss_sum += loss_val * len(batch_labels)
            num_pairs += len(batch_labels)

            if step % 2000 == 0:
                print("Average loss at step ", step, ": ", loss_sum / num_pairs)
        
        avg_loss = loss_sum / num_pairs

        return avg_loss 



def construct_graph(alpha, rho, bias, train_sidevec, train_inputs, train_labels, config, log_wcount):

    alpha_select = tf.gather(alpha, train_inputs, name='context_alpha')
    alpha_sum = tf.reduce_sum(alpha_select, 1, name='context_alpha_sum')
    embed = tf.concat(concat_dim=1, values=[alpha_sum, train_sidevec], name='context_vector')

    # Construct the variables for the NCE loss
    pos_rho = tf.gather(rho, train_labels, name='label_rho')
    pos_prod = tf.reduce_sum(tf.mul(embed, pos_rho), 1) + tf.gather(bias, tf.squeeze(train_labels))

    # bernoulli loss
    pos_loss = tf.nn.softplus(- pos_prod)
    pos_loss_mean = tf.reduce_mean(pos_loss)

    temp1 = tf.reduce_mean(train_inputs)
    temp2 = tf.reduce_mean(alpha_sum)


    # negative samples
    neg_dist = tf.contrib.distributions.Categorical(logits=log_wcount)
    shape = tf.concat(0, [[config['num_neg']], tf.shape(train_labels)])
    neg_label = neg_dist.sample(shape, name='negative_labels')
    neg_rho = tf.gather(rho, neg_label)
    neg_prod = tf.reduce_sum(tf.mul(embed, neg_rho), 2) + tf.gather(bias, tf.squeeze(train_labels))

    # bernoulli loss
    neg_loss = tf.nn.softplus(neg_prod)

    neg_loss_sum = tf.reduce_sum(tf.reduce_mean(neg_loss, 1))

    loss = pos_loss_mean + neg_loss_sum

    word_udist = tf.contrib.distributions.Categorical(logits=np.ones(len(log_wcount), dtype=np.float32))
    iword = word_udist.sample([1])
    regularizer = tf.reduce_mean(tf.square(tf.gather(alpha, iword))) + tf.reduce_mean(tf.square(tf.gather(rho, iword)))

    objective = loss + regularizer * config['reg_weight']

    return objective, loss


def generate_batch(review, config):

    text = review['text']
    atts = review['atts']
    
    half_window = config['half_window']
    span = 2 * half_window
    
    if len(text) <= span:
        raise Exception('Review is too short')

    batch_size = len(text) - span

    if config['use_sideinfo']:
        sideinfo = np.tile(atts, [batch_size, 1])
    else:
        sideinfo = np.zeros((batch_size, 0))

    label_ind = np.arange(half_window, batch_size + half_window)
    labels = text[label_ind]

    context_ind = np.r_[np.arange(0, half_window), np.arange(half_window + 1, 2 * half_window + 1)]
    context_ind = np.tile(context_ind, [batch_size, 1]) + np.arange(0, batch_size)[:, None]
    context = text[context_ind] 

    return sideinfo, context, labels
    



