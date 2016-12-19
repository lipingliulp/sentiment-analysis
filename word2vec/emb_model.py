from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections

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
        tf.set_random_seed(27)
        # Input data.
        dim_atts = len(reviews[0]['atts']) 
        train_sidevec = tf.placeholder(tf.float32, shape=[dim_atts], name='atts')

        train_inputs = tf.placeholder(tf.int32, shape=[None, 2 * config['half_window']], name='train_contexts')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        if init_model == None:
            weight = tf.Variable(tf.zeros([vocabulary_size, dim_atts]))
            alpha = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1, 1))
            rho = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1))

            #invmu = tf.constant(np.ones(vocabulary_size) * np.log(10000), dtype=tf.float32)
            invmu = tf.Variable(tf.ones(vocabulary_size) * 10)
        else:
            alpha  = tf.Variable(init_model['alpha'])
            invmu  = tf.Variable(init_model['invmu'])
            rho    = tf.Variable(init_model['rho'])
            weight = tf.Variable(init_model['weight'])

            print('use parameters of the initial model')

        objective, loss, temp = construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config, log_wcount)
        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.AdagradOptimizer(0.4).minimize(objective)

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
            rind = np.random.choice(len(reviews), 1)
            review = reviews[rind]
            batch_sidevec, batch_inputs, batch_labels = generate_batch(review, config)
            if config['use_sideinfo']:
                feed_dict = {train_sidevec : batch_sidevec, train_inputs : batch_inputs, train_labels : batch_labels}
            else:
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, tempval = session.run([optimizer, loss, temp], feed_dict=feed_dict)

            loss_logg.append(loss_val)
            average_loss += loss_val
            loss_count = loss_count + 1
            if np.isnan(loss_val):
                print('temp value is ', tempval)
                raise Exception('Loss value is nan')


            # print loss every 2000 iterations
            if step % 2000 == 0:
                if step > 0:
                  average_loss /= loss_count
                # The average loss is an estimate of the loss over the last 2000 batches.
                tempval = session.run(temp, feed_dict=feed_dict)

                print("Average loss at step ", step, ": ", average_loss, " tempval:", tempval)
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
   
        
        # save model parameters to dict
        model = dict(alpha=alpha.eval(), rho=rho.eval(), invmu=invmu.eval(), weight=weight.eval())

        return model, loss_logg



def evaluate_emb(reviews, model, config, voc_dict):

    embedding_size = config['K']
    vocabulary_size = len(voc_dict['dic'])
    wcount = [voc_dict['freq'][i][1] for i in xrange(vocabulary_size)]
    log_wcount = np.array(wcount, dtype=np.float32)
    log_wcount = 0.75 * np.log(log_wcount) 

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # Input data.
        dim_atts = len(reviews[0]['atts']) 
        train_sidevec = tf.placeholder(tf.float32, shape=[dim_atts], name='atts')

        train_inputs = tf.placeholder(tf.int32, shape=[None, 2 * config['half_window']], name='train_contexts')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')

        # Ops and variables pinned to the CPU because of missing GPU implementation
        alpha = tf.constant(model['alpha'])
        rho = tf.constant(model['rho'])
        invmu = tf.constant(model['invmu'])
        weight = tf.constant(model['weight'])

        objective, loss, word_loss = construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config, log_wcount)

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

            if step % 2000 == 0:
                print("Loss mean and std at step ", step, ": ", np.mean(np.concatenate(loss_array)), np.std(np.concatenate(loss_array)))
        

        return np.concatenate(loss_array)


def construct_exposure_graph(alpha, rho, invmu, weight, train_sidevec, train_inputs, train_labels, config, log_wcount):

    nneg = config['num_neg'] 
    neg_pos_ratio = config['negpos_ratio'] 

    # prepare embedding of context 
    alpha_select = tf.gather(alpha, train_inputs, name='context_alpha')
    embed = tf.reduce_sum(alpha_select, 1, name='context_alpha_sum')

    #positives: construct the variables
    pos_rho = tf.gather(rho, train_labels, name='label_rho')
    pos_logits = tf.reduce_sum(tf.mul(embed, pos_rho), 1)
    pos_invmu = tf.gather(invmu, train_labels)

    # elbo of positive samples. Note that q_ij = 1 
    if config['exposure']:
        pos_obj_pi = - tf.nn.softplus(- pos_logits)
        if config['use_sideinfo']:
            pos_score = tf.reduce_sum(tf.mul(tf.gather(weight, train_labels), train_sidevec), 1)
            pos_obj_mu = - tf.nn.softplus(- pos_invmu - pos_score)
        else: 
            pos_obj_mu = - tf.nn.softplus(- pos_invmu)
        pos_obj = tf.reduce_mean(pos_obj_pi) + tf.reduce_mean(pos_obj_mu)

        pos_word_obj = pos_obj_pi + pos_obj_mu
    else:
        pos_obj_pi = - tf.nn.softplus(- pos_logits) #- pos_invmu)
        pos_obj = tf.reduce_mean(pos_obj_pi) 
        
        pos_word_obj = pos_obj_pi

    # negative samples. # how to sample without replacement? 
    batch_size = tf.shape(train_inputs)[0]
    wdist = tf.contrib.distributions.Categorical(logits=log_wcount)
    neg_words = wdist.sample(nneg)
    neg_rho = tf.gather(rho, neg_words)
    neg_logits = tf.matmul(embed, neg_rho, transpose_b=True)
    
    neg_invmu = tf.gather(invmu, neg_words)

    # calculate posterior probabilities
    if config['exposure']:
        neg_pi = tf.nn.sigmoid(neg_logits)

        if config['use_sideinfo']:
            neg_score = tf.reduce_sum(tf.mul(tf.gather(weight, neg_words), train_sidevec), 1)
            neg_mu = tf.expand_dims(tf.sigmoid(neg_invmu + neg_score), 0)
        else: 
            neg_mu = tf.expand_dims(tf.sigmoid(neg_invmu), 0)

        neg_mupi = tf.mul(neg_mu, neg_pi)
        floppy_loss = tf.log(1 - neg_mupi)
    else:
        neg_logitsb = neg_logits # + tf.expand_dims(neg_invmu, 0)
        floppy_loss = - tf.nn.softplus(neg_logitsb)

    mask = tf.not_equal(tf.tile(tf.expand_dims(neg_words, 0), [batch_size, 1]), tf.expand_dims(train_labels, 1))
    mask = tf.cast(mask, tf.float32)

    # calculate elbo for negative samples
    neg_word_obj = tf.reduce_sum(tf.mul(floppy_loss, mask), 1)  / tf.reduce_sum(mask, 1)
    neg_obj = tf.reduce_sum(tf.mul(floppy_loss, mask)) / tf.reduce_sum(mask)


    # calculate regularizer
    regularizer = (tf.reduce_mean(tf.square(neg_rho)) + tf.reduce_mean(tf.square(alpha_select))) * tf.cast(tf.shape(alpha)[1], tf.float32)

    # calculate the final objective
    loss = - pos_obj - neg_pos_ratio * neg_obj
    objective = regularizer * config['reg_weight'] + loss
    #objective = loss
    
    word_loss = - pos_word_obj - neg_word_obj

    return objective, loss, word_loss


def generate_batch(review, config):

    text = review['text']
    atts = review['atts']
    
    half_window = config['half_window']
    span = 2 * half_window
    
    if len(text) <= span:
        raise Exception('Review is too short')

    batch_size = len(text) - span

    if config['use_sideinfo']:
        sideinfo = atts
    else:
        sideinfo = np.zeros((batch_size, 0))

    label_ind = np.arange(half_window, batch_size + half_window)
    labels = text[label_ind]

    context_ind = np.r_[np.arange(0, half_window), np.arange(half_window + 1, 2 * half_window + 1)]
    context_ind = np.tile(context_ind, [batch_size, 1]) + np.arange(0, batch_size)[:, None]
    context = text[context_ind] 

    return sideinfo, context, labels

# draw negative samples from the multinomial distribution
# labels: a tensor
# nneg: number of negative samples, integer
# logits: numpy vector, log p + const 

# return negative samples 
def negative_sample(labels, nneg, logits, session=None):
    
    extra = 30
   
    wdist = tf.contrib.distributions.Categorical(logits=logits)
    batch_size = tf.shape(labels)[0]
    ws = wdist.sample([batch_size, nneg + extra])


    neq = tf.not_equal(ws, tf.expand_dims(labels, 1))
    neq = tf.cast(neq, tf.int32)
    ind = tf.cumsum(neq, axis=1) + (1 - neq) * (nneg + extra)

    mask = tf.less_equal(ind, nneg) 
    mask = tf.reshape(mask, [batch_size, nneg + extra])

    negs = tf.boolean_mask(ws, mask)
    
    samples = tf.reshape(negs, (batch_size, nneg)) 

    return samples


