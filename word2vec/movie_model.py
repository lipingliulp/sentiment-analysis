from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
from scipy import sparse
import sys

def separate_valid(reviews):
    frac = 0.1
    review_size = reviews['scores'].shape[0]
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)

    trainset = dict(scores=reviews['scores'][tind, :], atts=reviews['atts'][tind])
    validset = dict(scores=reviews['scores'][vind, :], atts=reviews['atts'][vind])
    
    return trainset, validset
    





def get_problem_sizes(reviews, config):
    review_size = reviews['scores'].shape[0]
    movie_size = reviews['scores'].shape[1]
    dim_atts = reviews['atts'].shape[1]
    
    return review_size, movie_size, dim_atts

# config = dict(K=K, voc_size=voc_size, use_sideinfo=use_sideinfo, half_window=half_window)
def fit_emb(reviews, config, init_model):

    reviews, valid_reviews = separate_valid(reviews)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)

        inputs, outputs, model_param = construct_model_graph(reviews, config, init_model, training=True)

        optimizer = tf.train.AdagradOptimizer(0.1).minimize(outputs['objective'])
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_logg = np.zeros([config['max_iter'], 2]) 
        review_size = reviews['scores'].shape[0]
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            atts, indices, labels = generate_batch(reviews, rind)
            feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}

            _, llh_val, obj_val, debug_val = session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv']), feed_dict=feed_dict)
            loss_logg[step - 1, :] = np.array([llh_val, obj_val])

            # print loss every 2000 iterations
            nprint = 5000
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                

                valid_size = valid_reviews['scores'].shape[0]
                llh_sum = 0
                for iv in xrange(valid_size): 
                    atts, indices, labels = generate_batch(valid_reviews, iv)
                    feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}
                    llh_val = session.run((outputs['llh']), feed_dict=feed_dict)
                    llh_sum = llh_sum + llh_val
  
                valid_llh = llh_sum / valid_size
                
                avg_val = np.mean(loss_logg[step - nprint : step], axis=0)
                print("iteration[", step, "]: average llh and obj are ", avg_val, ' validation llh is ', valid_llh)
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    debug_val = session.run(outputs['debugv'], feed_dict=feed_dict)
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
    
        # save model parameters to dict
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                     invmu=model_param['invmu'].eval(), 
                    weight=model_param['weight'].eval())

        return model, loss_logg

def evaluate_emb(reviews, model, config):

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        inputs, outputs, model_param = construct_model_graph(reviews, config, model, training=False)
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_array = [] 
        review_size = reviews['scores'].shape[0]
        for step in xrange(review_size):
            att, index, label = generate_batch(reviews, step)
            feed_dict = {inputs['input_att']: att, inputs['input_ind']: index, inputs['input_label']: label}
            ins_llh_val = session.run(outputs['ins_llh'], feed_dict=feed_dict)
            loss_array.append(ins_llh_val)
            
        print("Loss mean and std: ", np.mean(np.concatenate(loss_array)), np.std(np.concatenate(loss_array)))
        
        return loss_array

def logprob_zero(context_emb, rho, input_ind, input_att, weight, invmu, config, input_label, training):

    movie_size = int(rho.get_shape()[0])
    flag = tf.Variable(tf.ones(movie_size, dtype=tf.bool))
    flag = tf.scatter_update(flag, np.arange(movie_size), np.tile([True], movie_size))
    flag = tf.scatter_update(flag, input_ind, tf.tile([False], tf.shape(input_ind)))
    sind = tf.cast(tf.squeeze(tf.where(flag)), tf.int32)

    if training:
        print(movie_size)
        nsample = tf.cast(config['sample_ratio'] * tf.cast(tf.shape(sind)[0], dtype=tf.float32), tf.int32)
        sind = tf.gather(tf.random_shuffle(sind), tf.range(nsample))

    rho_z = tf.gather(rho, sind) 

    # poisson distribution
    #lamb_z = tf.nn.softplus(tf.reduce_sum(rho_z * context_emb, reduction_indices=1)) + 1e-6
    #logprob_z = - lamb_z 

    # binomial distribution
    rlog = tf.reduce_sum(rho_z * context_emb, reduction_indices=1)
    logprob_z  = - 3.0 * tf.nn.softplus(rlog)

    if config['exposure']:
        logits = tf.gather(invmu, sind)
        if config['use_sideinfo']:
            logits = logits + tf.reduce_sum(tf.gather(weight, sind) * tf.expand_dims(input_att, 0), 1)
        
        log_nobs_prob = - tf.nn.softplus(logits) 
        log_obs_prob = - tf.nn.softplus(-logits) 
        
        logprob = logsumexp(log_obs_prob + logprob_z, log_nobs_prob)

    else:
        logprob = logprob_z

    if training:
        noisy_logprob_sum = tf.reduce_sum(logprob) * tf.cast(movie_size, tf.float32) / tf.cast(tf.shape(sind)[0], tf.float32)
    else:
        noisy_logprob_sum = tf.reduce_sum(logprob)
    
    return noisy_logprob_sum, logprob, sind, [tf.reduce_min(logprob_z), tf.reduce_max(logprob_z)]

def gammaln(x):
    # fast approximate gammaln from paul mineiro
    # http://www.machinedlearnings.com/2011/06/faster-lda.html
    logterm = tf.log (x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)

def logprob_nonz(alpha_emb, rho_select, input_ind, input_att, weight, invmu, rate, config, training=True):

    # poisson distribuiton
    #lamb_nz = tf.nn.softplus(tf.reduce_sum(rho_select * alpha_emb, reduction_indices=1)) + 1e-6
    #logprob_nz = rate * tf.log(lamb_nz) - lamb_nz - gammaln(rate + 1.0)

    # binomial distribution
    rlog = tf.reduce_sum(rho_select * alpha_emb, reduction_indices=1)
    logminusprob = - tf.nn.softplus(rlog)
    logplusprob = - tf.nn.softplus(- rlog)
    logprob_nz = np.log(6.0) - gammaln(rate + 1.0) - gammaln(4.0 - rate) + rate * logplusprob + (3.0 - rate) * logminusprob

    if config['exposure']:
        logits = tf.gather(invmu, input_ind)
        if config['use_sideinfo']:
            logits = logits + tf.reduce_sum(tf.gather(weight, input_ind) * tf.expand_dims(input_att, 0), 1)
        
        log_obs_prob = - tf.nn.softplus(- logits) 
        
        logprob = log_obs_prob + logprob_nz

    else:
        logprob = logprob_nz
    
    logprob_sum = tf.reduce_sum(logprob)
    
    return logprob_sum, logprob, []

def construct_model_graph(reviews, config, init_model=None, training=True):

    embedding_size = config['K']
    review_size, movie_size, dim_atts = get_problem_sizes(reviews, config)


    input_att = tf.placeholder(tf.float32, shape=[dim_atts])
    input_ind = tf.placeholder(tf.int32, shape=[None])
    input_label = tf.placeholder(tf.int32, shape=[None])

    if training: 
        if init_model == None:
            weight = tf.Variable(tf.zeros([movie_size, dim_atts]))
            alpha = tf.Variable(tf.random_uniform([movie_size, embedding_size], -1, 1))
            rho = tf.Variable(tf.truncated_normal([movie_size, embedding_size],stddev=1))
            invmu = tf.Variable(tf.random_uniform([movie_size], -1, 1))
        else:
            alpha  = tf.Variable(init_model['alpha'])
            invmu  = tf.Variable(init_model['invmu'])
            rho    = tf.Variable(init_model['rho'])
            weight = tf.Variable(init_model['weight'])
            print('use parameters of the initial model')
    else: 
        alpha  = tf.constant(init_model['alpha'])
        invmu  = tf.constant(init_model['invmu'])
        rho    = tf.constant(init_model['rho'])
        weight = tf.constant(init_model['weight'])


    # rate 
    rate = tf.cast(input_label, tf.float32)
    nnz = tf.cast(tf.shape(rate)[0], tf.float32)

    #prepare embedding of context 
    alpha_select = tf.gather(alpha, input_ind, name='context_alpha')
    alpha_weighted = alpha_select * tf.expand_dims(rate, 1)
    alpha_sum = tf.reduce_sum(alpha_weighted, keep_dims=True, reduction_indices=0)

    z_logp, z_insprob, sind, _ = logprob_zero(alpha_sum / nnz, rho, input_ind, input_att, weight, invmu, config, input_label, training=training)

    alpha_emb = (alpha_sum - alpha_select) / (nnz - 1) 
    rho_select = tf.gather(rho, input_ind)
    nz_logp, nz_insprob, debug = logprob_nonz(alpha_emb, rho_select, input_ind, input_att, weight, invmu, rate, config, training=training)

    llh = z_logp + nz_logp
    if training:
        ins_llh = None
    else:
        ins_logprob = tf.concat(0, [z_insprob, nz_insprob])
        ins_ind = tf.concat(0, [sind, input_ind])
        ins_llh = tf.scatter_update(tf.Variable(tf.zeros(movie_size)), ins_ind, ins_logprob)

    # negative samples. # how to sample without replacement? 
    regularizer = (tf.reduce_sum(tf.square(rho_select)) + tf.reduce_sum(tf.square(alpha_select))) \
                  * (0.5 * movie_size / (nnz * float(review_size)))
    
    if config['use_sideinfo']:
        wreg = 100000.0 * tf.reduce_sum(tf.square(tf.gather(weight, input_ind))) \
               * (0.5 * movie_size / (nnz * float(review_size)))
        regularizer = regularizer + wreg

    objective = regularizer * config['reg_weight'] - llh 

    inputs = {'input_att': input_att, 'input_ind': input_ind, 'input_label': input_label} 
    outputs = {'objective': objective, 'llh': llh, 'ins_llh': ins_llh, 'debugv': debug}
    model_param = {'alpha': alpha, 'rho': rho, 'weight': weight, 'invmu': invmu}

    return inputs, outputs, model_param 

def logsumexp(vec1, vec2):
    flag = tf.greater(vec1, vec2)
    maxv = tf.select(flag, vec1, vec2)
    lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
    return lse

def generate_batch(reviews, rind):
    atts = reviews['atts'][rind, :]
    _, ind, rate = sparse.find(reviews['scores'][rind, :])
    return atts, ind, rate 

