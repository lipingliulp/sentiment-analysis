from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections

import sys


# config = dict(K=K, voc_size=voc_size, use_sideinfo=use_sideinfo, half_window=half_window)
def fit_emb(reviews, config, init_model):

    embedding_size = config['K']
    movie_size = reviews.shape[1]

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # Input data.
        dim_atts = 0 
        train_sidevec = tf.placeholder(tf.float32, shape=[dim_atts])
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None])

        if init_model == None:
            weight = tf.Variable(tf.zeros([movie_size, dim_atts]))
            alpha = tf.Variable(
                tf.random_uniform([movie_size, embedding_size], -1, 1))
            rho = tf.Variable(
                tf.truncated_normal([movie_size, embedding_size],
                                stddev=1))

            invmu = tf.Variable(tf.ones(movie_size) * 10)
        else:
            alpha  = tf.Variable(init_model['alpha'])
            invmu  = tf.Variable(init_model['invmu'])
            rho    = tf.Variable(init_model['rho'])
            weight = tf.Variable(init_model['weight'])

            print('use parameters of the initial model')

        objective, loss, word_loss, temp = construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config)
        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.AdagradOptimizer(0.8).minimize(objective)

        # Add variable initializer.
        init = tf.initialize_all_variables()
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        loss_logg = []
        average_loss = 0
        loss_count = 0 
        for step in xrange(1, config['max_iter'] + 1):
            rind = np.random.choice(len(reviews))
            review = reviews[rind]
            batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
            if config['use_sideinfo']:
                feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}
            else:
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

            loss_logg.append(loss_val)
            average_loss += loss_val
            loss_count = loss_count + 1
            # print loss every 2000 iterations
            if step % 5000 == 0:
                if step > 0:
                  average_loss /= loss_count
                # The average loss is an estimate of the loss over the last 2000 batches.
                tempval = session.run(temp, feed_dict=feed_dict)

                print("Average loss at step ", step, ": ", average_loss, " tempval:", tempval)
                average_loss = 0
                loss_count = 0
    
        # save model parameters to dict
        model = dict(alpha=alpha.eval(), rho=rho.eval(), invmu=invmu.eval(), weight=weight.eval())

        return model, loss_logg

def evaluate_emb(reviews, model, config):

    embedding_size = config['K']
    movie_size = reviews.shape[1]

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # Input data.
        dim_atts = len(reviews[0]['atts']) 
        train_sidevec = tf.placeholder(tf.float32, shape=[dim_atts], name='atts')

        train_inputs = tf.placeholder(tf.int32, shape=[None], name='train_contexts')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')

        # Ops and variables pinned to the CPU because of missing GPU implementation
        alpha = tf.constant(model['alpha'])
        rho = tf.constant(model['rho'])
        invmu = tf.constant(model['invmu'])
        weight = tf.constant(model['weight'])

        objective, loss, word_loss, _ = construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config)

        # Add variable initializer.
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        loss_array = [] 
        for step in xrange(len(reviews)):
            review = reviews[step % len(reviews)]
            batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
            if config['use_sideinfo']:
                feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}
            else:
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
    
            word_loss_val = session.run(word_loss, feed_dict=feed_dict)
            loss_array.append(word_loss_val)

            if step % 5000 == 0:
                print("Loss mean and std at step ", step, ": ", np.mean(np.concatenate(loss_array)), np.std(np.concatenate(loss_array)))
        
        return loss_array


def construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config):

    #prepare embedding of context 
    alpha_select = tf.gather(alpha, train_inputs, name='context_alpha')
    alpha_sum = tf.reduce_sum(alpha_select, keep_dims=True, reduction_indices=0)
    embed = alpha_sum - alpha_select   

    #positives: construct the variables
    rho_select = tf.gather(rho, train_inputs, name='label_rho')
    prod = tf.reduce_sum(embed * rho_select, reduction_indices=1)

    lamb = tf.nn.softplus(prod)
    rate = train_labels - 1
    logprob = - lamb + tf.cast(rate, tf.float32) * tf.log(lamb)

    # elbo of positive samples. Note that q_ij = 1 
    if config['exposure']:

        if config['use_sideinfo']:
            raise Exception('Not implemented')
        else: 
            obs_logits = invmu

        obs_logprob = - tf.nn.softplus(- obs_logits)
        obs_joint = obs_logprob + logprob

        flag = tf.equal(rate, 0)
        zobs_logprob = - tf.nn.softplus(tf.boolean_select(obs_logits, flag))
        joint_select = tf.boolean_select(obs_joint, flag)

        marg = tf.logsumexp(zobs_logprob, joint_select)

        llh = tf.reduce_sum(marg) + tf.reduce_sum(tf.boolean_select(obs_joint, tf.logical_not(flag)))

        find = tf.squeeze(tf.where(flag))
        ins_llh = tf.scatter_update(obs_joint, find, marg) 
        
    else:
        llh = tf.reduce_sum(logprob)
        ins_llh = logprob

    # negative samples. # how to sample without replacement? 
    regularizer = (tf.reduce_sum(tf.square(rho_select)) + tf.reduce_sum(tf.square(alpha_select))) * 0.5

    objective = regularizer * config['reg_weight'] - llh 

    return objective, llh, ins_llh, llh

def logsumexp(vec1, vec2):
    flag = tf.greater(vec1, vec2)
    maxv = tf.select(flag, vec1, vec2)
    lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
    return lse

def generate_batch(review, config):

    
    ind = review.nonzero()[0] 
    rate = review[ind]

    batch_size = ind.shape[0]
    if config['use_sideinfo']:
        sideinfo = np.zeros((batch_size, 0))
    else:
        sideinfo = np.zeros((batch_size, 0))

    return sideinfo, ind, rate 

